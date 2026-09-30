"""工具网关：所有 Agent 工具的统一执行入口。

每个工具声明自己的策略（ToolPolicy），网关按同一条链执行：
  校验入参 → 熔断检查 → 超时内执行 → 校验返回值 → 限制结果大小 → 记录指标和统计

纯本地计算的工具（报价、退款）默认不开熔断：没有外部依赖，出错只可能是代码 bug，
熔断解决不了；它们仍然有超时、入参和返回值校验、指标。依赖 Redis 等外部服务的写入类工具
开熔断，连续失败后快速失败，不拖慢整个对话。
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from tooling.breaker import CircuitBreaker

logger = logging.getLogger(__name__)

SIDE_EFFECTS = ("read", "write", "external")


@dataclass(frozen=True)
class ToolPolicy:
    """一个工具的执行策略。"""

    timeout_s: float = 5.0
    side_effect: str = "read"         # read 只读 / write 写入 / external 访问外部服务
    breaker: bool = False             # 是否启用熔断（有外部依赖的工具才需要）
    max_result_chars: int = 12000     # 返回给模型的 JSON 最大长度，超出就截断并标记

    def __post_init__(self) -> None:
        if self.side_effect not in SIDE_EFFECTS:
            raise ValueError(f"side_effect 必须是 {SIDE_EFFECTS} 之一")


@dataclass
class ToolOutcome:
    """一次工具调用的结果。ok 表示处理函数正常跑完（结果里的 success 是业务层面的成败）。"""

    result: Any
    ok: bool
    kind: str                         # ok / invalid_args / invalid_output / timeout / error / circuit_open
    error: str = ""
    latency_ms: float = 0.0
    truncated: bool = False
    result_success: Optional[bool] = None


@dataclass
class _ToolStat:
    calls: int = 0
    by_kind: Dict[str, int] = field(default_factory=dict)
    total_ms: float = 0.0
    max_ms: float = 0.0


def validate_args(spec: Any, args: Any) -> Dict[str, Any]:
    """校验入参并返回交给处理函数的字典；不合格抛 ValueError。"""
    if not isinstance(args, dict):
        raise ValueError("工具参数必须是 JSON 对象")
    model = getattr(spec, "input_model", None)
    if model is not None:
        try:
            return model.model_validate(args).model_dump(exclude_none=True)
        except ValidationError as ex:
            raise ValueError(_format_validation_error(ex)) from None
    return _legacy_validate(spec, args)


def _format_validation_error(ex: ValidationError) -> str:
    """把 Pydantic 的错误翻成模型能看懂、能据此改正的中文。"""
    parts = []
    for err in ex.errors()[:5]:
        where = ".".join(str(p) for p in err["loc"]) or "参数"
        if err["type"] == "extra_forbidden":
            parts.append(f"不允许的工具参数: {where}")
        elif err["type"] == "missing":
            parts.append(f"缺少必需参数: {where}")
        else:
            parts.append(f"{where}: {err['msg']}")
    return "；".join(parts)


def _legacy_validate(spec: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    """没有 Pydantic 模型的工具沿用的简化校验：必需参数、未知参数、基本类型。"""
    schema = spec.input_schema
    for name in schema.get("required", []):
        if name not in args:
            raise ValueError(f"缺少必需参数: {name}")
    properties = schema.get("properties", {})
    unknown = set(args) - set(properties)
    if unknown and schema.get("additionalProperties") is False:
        raise ValueError(f"不允许的工具参数: {', '.join(sorted(unknown))}")
    type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool}
    for key, value in args.items():
        expected = properties.get(key, {}).get("type")
        if expected in type_map and not isinstance(value, type_map[expected]):
            raise ValueError(f"参数 {key} 类型错误，期望 {expected}")
    return dict(args)


def shrink_result(result: Any, max_chars: int) -> tuple[Any, bool]:
    """结果太长时，反复把最长的列表减半，并标明被截断，不让模型把部分数据当成全部。"""
    def size(obj: Any) -> int:
        return len(json.dumps(obj, ensure_ascii=False, default=str))

    if not isinstance(result, dict) or size(result) <= max_chars:
        return result, False
    out = dict(result)
    dropped: Dict[str, int] = dict(out.get("truncated_lists", {}))
    for _ in range(12):
        lists = [(k, v) for k, v in out.items() if isinstance(v, list) and len(v) > 1]
        if not lists or size(out) <= max_chars:
            break
        key, items = max(lists, key=lambda kv: len(json.dumps(kv[1], ensure_ascii=False, default=str)))
        dropped[key] = dropped.get(key, len(items))
        out[key] = items[: max(1, len(items) // 2)]
    out["truncated"] = True
    out["truncated_lists"] = {k: {"total": n, "returned": len(out[k])} for k, n in dropped.items()}
    if size(out) > max_chars:                      # 没有可减的列表，退回直接截字符串
        text = json.dumps(out, ensure_ascii=False, default=str)[:max_chars]
        return {"success": bool(result.get("success", True)), "truncated": True, "partial": text}, True
    return out, True


class ToolGateway:
    """进程内唯一的工具执行入口，持有每个工具的熔断器和统计。"""

    def __init__(self) -> None:
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._stats: Dict[str, _ToolStat] = {}

    def _breaker(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker()
        return self._breakers[name]

    async def call(self, spec: Any, req: Any, args: Any) -> ToolOutcome:
        policy: ToolPolicy = getattr(spec, "policy", ToolPolicy())
        name = spec.name
        t0 = time.monotonic()

        def finish(kind: str, result: Any, error: str = "", *, ok: bool = False, truncated: bool = False) -> ToolOutcome:
            ms = (time.monotonic() - t0) * 1000
            self._record(name, kind, ms)
            return ToolOutcome(
                result=result, ok=ok, kind=kind, error=error, latency_ms=ms, truncated=truncated,
                result_success=bool(result["success"]) if isinstance(result, dict) and "success" in result else None,
            )

        try:
            data = validate_args(spec, args)
        except ValueError as ex:
            return finish("invalid_args", {"success": False, "error": str(ex)}, str(ex))

        breaker = self._breaker(name) if policy.breaker else None
        if breaker is not None and not breaker.allow():
            msg = f"工具 {name} 暂时不可用（连续失败，熔断中），请稍后再试或转顾问处理"
            return finish("circuit_open", {"success": False, "error": msg}, msg)

        try:
            result = await asyncio.wait_for(self._invoke(spec, req, data), timeout=policy.timeout_s)
        except asyncio.TimeoutError:
            if breaker is not None:
                breaker.record_failure()
            msg = f"工具 {name} 执行超过 {policy.timeout_s:g} 秒"
            logger.warning(msg)
            return finish("timeout", {"success": False, "error": msg}, msg)
        except Exception as ex:                     # noqa: BLE001 —— 工具出错不能拖垮整个请求
            if breaker is not None:
                breaker.record_failure()
            logger.warning("Agent 工具 %s 执行失败: %s", name, ex)
            return finish("error", {"success": False, "error": str(ex)}, str(ex))

        if breaker is not None:
            breaker.record_success()

        output_model = getattr(spec, "output_model", None)
        if output_model is not None and isinstance(result, dict):
            try:
                output_model.model_validate(result)
            except ValidationError as ex:
                msg = f"工具 {name} 的返回值没有通过校验：{_format_validation_error(ex)}"
                logger.error(msg)
                return finish("invalid_output", {"success": False, "error": "工具返回的结果没有通过内部校验，请转顾问核算"}, msg)

        result, truncated = shrink_result(result, policy.max_result_chars)
        return finish("ok", result, ok=True, truncated=truncated)

    @staticmethod
    async def _invoke(spec: Any, req: Any, data: Dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(spec.handler):
            return await spec.handler(req, data)
        result = await asyncio.to_thread(spec.handler, req, data)   # 同步函数放进线程，超时才真的能生效
        if inspect.isawaitable(result):
            result = await result
        return result

    def record_rejected(self, kind: str) -> None:
        """模型调用了不在白名单里的工具：只记指标，工具名是模型生成的，不能当标签。"""
        try:
            from core.metrics import TOOL_CALLS
            TOOL_CALLS.labels(tool="__unknown__", outcome=kind).inc()
        except Exception:                           # noqa: BLE001
            logger.debug("工具指标记录失败", exc_info=True)

    def _record(self, name: str, kind: str, ms: float) -> None:
        stat = self._stats.setdefault(name, _ToolStat())
        stat.calls += 1
        stat.by_kind[kind] = stat.by_kind.get(kind, 0) + 1
        stat.total_ms += ms
        stat.max_ms = max(stat.max_ms, ms)
        try:
            from core.metrics import TOOL_CALLS, TOOL_LATENCY
            TOOL_CALLS.labels(tool=name, outcome=kind).inc()
            TOOL_LATENCY.labels(tool=name).observe(ms / 1000)
        except Exception:                           # noqa: BLE001 —— 指标失败不影响调用
            logger.debug("工具指标记录失败", exc_info=True)

    def stats(self) -> List[Dict[str, Any]]:
        rows = []
        for name, stat in sorted(self._stats.items()):
            breaker = self._breakers.get(name)
            rows.append({
                "tool": name,
                "calls": stat.calls,
                "by_outcome": dict(stat.by_kind),
                "avg_ms": round(stat.total_ms / stat.calls, 2) if stat.calls else 0.0,
                "max_ms": round(stat.max_ms, 2),
                "circuit": breaker.state.value if breaker else "n/a",
            })
        return rows


_default_gateway = ToolGateway()


def get_gateway() -> ToolGateway:
    return _default_gateway
