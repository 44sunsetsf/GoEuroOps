"""
亮点：检索工具治理层（知识库检索的缓存、熔断、降级与重排）

核心问题：工具调用出错（检索不全、召回不好）怎么优化？

本模块的答案：
  1. 查询改写（Query Rewriting）—— 用 LLM 把用户原始问题扩写成多个角度的子查询，
     再合并去重，解决"召回不全"问题。
  2. 结果重排（Reranking）—— 用 bge-reranker（cross-encoder，retrieval/rerank.py）给召回结果
     打 0–1 的相关度，重新排序并过滤不相关的，解决"召回不好/排序差"问题。
  3. 熔断器（Circuit Breaker）—— 连续失败超阈值时自动断开，防止雪崩。
  4. 结果缓存（TTL Cache）—— 相同参数直接返回缓存，减少重复调用。
  5. 降级策略（Fallback）—— 工具不可用时返回有意义的降级结果。
"""
import asyncio
import hashlib
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.rag_gate import retrieval_query
from retrieval.rerank import ApiReranker, passes_threshold
from tooling.breaker import CircuitBreaker

from core.llm_utils import NO_THINKING_KWARGS, extract_text_content
from core.llm_utils import make_client, safe_text
from core.config import DEFAULT_MODEL

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    success:        bool
    data:           Any
    tool_name:      str
    error:          Optional[str] = None
    cached:         bool = False
    latency_ms:     float = 0.0
    reranked:       bool = False   # 是否经过重排


@dataclass
class ToolStats:
    """工具运行时统计，供 Monitor 读取。"""
    total:              int = 0
    success:            int = 0
    failed:             int = 0
    total_latency_ms:   float = 0.0
    consecutive_fails:  int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0


# ── 熔断器 ────────────────────────────────────────────────────────────────────

# ── 工具定义 ──────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name:        str
    description: str
    handler:     Callable                    # async (params, context) -> Any
    schema:      Dict[str, Any]              # JSON Schema
    cache_ttl:   float = 0.0                 # 0 = 不缓存
    timeout_s:   float = 30.0
    supports_rerank: bool = False            # 是否支持结果重排
    fallback:    Optional[Callable] = None    # sync/async (params, context, error) -> Any

    # 运行时状态（不参与构造）
    stats:   ToolStats    = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)


# ── 检索工具管理器 ────────────────────────────────────────────────────────────

class RetrievalManager:
    """
    检索工具的治理层：缓存、超时、熔断、降级，以及查询改写和 LLM 重排。

    核心优化链路（针对检索类工具）：
      用户查询 → 查询改写（多角度子查询）→ 并行召回 → 结果重排 → 返回 Top-K
    """

    RERANK_CANDIDATES = 8        # 交给重排的候选文档数

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = DEFAULT_MODEL,
                 reranker: Optional[ApiReranker] = None):
        self._client = make_client(api_key, base_url, "tools")
        self._model  = model
        self._reranker = reranker
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, tuple] = {}   # key → (result, expire_at)

    # ── 注册 / 注销 ───────────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ── 核心调用 ──────────────────────────────────────────────────────────────

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
    ) -> ToolResult:
        """
        调用工具，完整执行链：
          缓存检查 → 熔断检查 → 参数校验 → 执行（含超时）→ 缓存写入
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, data=None, tool_name=name, error=f"工具不存在: {name}")

        # 缓存命中
        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params)
            if cached is not None:
                tool.stats.total += 1
                tool.stats.success += 1
                return ToolResult(success=True, data=cached, tool_name=name, cached=True)

        # 熔断检查
        ticket = tool.breaker.acquire()
        if ticket is None:
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, context, error)

        t0 = time.monotonic()
        tool.stats.total += 1
        try:
            # 参数校验（根据 JSON Schema 的 required 和 properties.type）
            self._validate_params(tool, params)

            data = await asyncio.wait_for(self._run_handler(tool, params, context), timeout=tool.timeout_s)
            latency = (time.monotonic() - t0) * 1000

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.total_latency_ms += latency
            tool.breaker.record_success(ticket)

            if tool.cache_ttl > 0:
                self._set_cache(name, params, data, tool.cache_ttl)

            return ToolResult(success=True, data=data, tool_name=name, latency_ms=latency)

        except asyncio.TimeoutError:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure(ticket)
            logger.error(f"工具超时: {name} ({tool.timeout_s}s)")
            return await self._fallback_result(tool, params, context, "执行超时")

        except Exception as ex:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure(ticket)
            logger.error(f"工具异常: {name} — {ex}")
            return await self._fallback_result(tool, params, context, str(ex))

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
    ) -> ToolResult:
        """工具不可用时返回降级结果，而不是把空错误直接暴露给调用方。"""
        if tool.fallback is None:
            return ToolResult(success=False, data=None, tool_name=tool.name, error=error)
        try:
            data = tool.fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            return ToolResult(
                success=True,
                data=data,
                tool_name=tool.name,
                error=error,
            )
        except Exception as ex:
            logger.error(f"工具降级失败: {tool.name} — {ex}")
            return ToolResult(success=False, data=None, tool_name=tool.name, error=f"{error}; fallback失败: {ex}")

    async def _run_handler(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """
        执行工具 handler。

        优先支持 async handler；如果历史工具仍是同步函数，则放入线程池执行，
        避免阻塞事件循环。
        """
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(params, context)
        result = await asyncio.to_thread(tool.handler, params, context)
        if inspect.isawaitable(result):
            return await result
        return result

    # ── 查询改写（解决召回不全）────────────────────────────────────────────────

    async def rewrite_query(self, query: str, n: int = 3, previous: str = "") -> List[str]:
        """
        用 LLM 将原始查询改写为 n 个不同角度的子查询。

        目的：单一查询往往只能召回某一角度的文档，
        多角度子查询并行检索后合并，显著提升召回率。

        示例：
          原始: "退款流程"
          改写: ["如何申请退款", "退款需要多少天", "退款政策是什么"]
        """
        prompt = f"""将以下用户查询改写为 {n} 个不同角度的搜索子查询，用于检索知识库。
要求：每个子查询角度不同，覆盖原始问题的不同方面。
原始查询: "{query}"
{f'上一轮用户说的是: "{previous}"。原始查询省略了主语或指代上一轮时，子查询要写完整（例如“那德国呢”写成“德国 CS 硕士申请截止时间”）。' if previous else ''}
返回 JSON 数组，例如: ["子查询1", "子查询2", "子查询3"]"""
        prompt = self._clean_text(prompt)
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                **NO_THINKING_KWARGS,
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("["), raw.rfind("]") + 1
            queries = json.loads(raw[s:e])
            # 原始查询也保留，去重
            return list(dict.fromkeys([query] + queries))
        except Exception as ex:
            logger.warning(f"查询改写失败，使用原始查询: {ex}")
            return [query]

    async def search_with_rewrite(
        self,
        tool_name: str,
        query: str,
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> ToolResult:
        """
        完整的检索优化链路：查询改写 → 并行召回 → 去重 → 重排（含相关性过滤）→ Top-K

        这是解决"检索不全、召回不好"的完整方案。
        domain 非空时只在该 Agent 的知识范围内检索，避免跨业务线串场。
        history 用来补全“那德国呢”这类追问：改写时参考上一轮用户的话，重排按补全后的问题打分。
        """
        full_query = retrieval_query(query, history)
        previous = full_query[: -len(query)].strip() if full_query != query.strip() else ""

        # 1. 查询改写：生成多角度子查询
        sub_queries = await self.rewrite_query(query, n=3, previous=previous)
        logger.info(f"查询改写: {query!r} → {sub_queries}")

        # 2. 并行召回：所有子查询同时检索
        recall_k = max(top_k, 5)
        tasks = [
            self.call(tool_name, {"query": q, "top_k": recall_k, "domain": domain}, context, use_cache=True)
            for q in sub_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. 合并去重：同一篇文档被不同子查询召回时只留一条（分数会不同，不能按整条结果去重），
        #    重复命中时保留分数更高的那一条。
        best_by_key: Dict[str, Dict[str, Any]] = {}
        degraded_error = ""
        for r in results:
            if isinstance(r, ToolResult) and r.success and isinstance(r.data, list):
                for item in r.data:
                    if not isinstance(item, dict):
                        continue
                    if item.get("fallback"):
                        # 降级提示不是检索到的知识，不能当证据交给模型
                        degraded_error = r.error or str(item.get("error") or "")
                        continue
                    key = str(item.get("doc_id") or f"{item.get('title', '')}\x1f{item.get('content', '')}")
                    existing = best_by_key.get(key)
                    if existing is None or float(item.get("score", 0) or 0) > float(existing.get("score", 0) or 0):
                        best_by_key[key] = item
        merged = list(best_by_key.values())

        if not merged:
            if degraded_error:
                return ToolResult(success=False, data=[], tool_name=tool_name,
                                  error=f"知识库暂时不可用（{degraded_error}），请稍后重试或请顾问确认")
            return ToolResult(success=False, data=[], tool_name=tool_name, error="所有子查询均无结果")

        # 4. 重排（按原问题打分）→ 过滤不够相关的 → Top-K；重排不可用时按混合分和原阈值
        ranked, reranked = await self._rerank(full_query, merged)
        kept = [item for item in ranked if passes_threshold(item)][:top_k]
        if not kept:
            return ToolResult(success=False, data=[], tool_name=tool_name, error="知识库中没有与问题足够相关的内容",
                              reranked=reranked)
        return ToolResult(success=True, data=kept, tool_name=tool_name, reranked=reranked)

    async def search_fast(
        self,
        tool_name: str,
        query: str,
        top_k: int = 3,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """快速检索：单次向量召回 + 重排，不做 LLM 改写。

        给意图门控 RAG 的投机预取使用——它和意图识别（一两秒）并行跑，重排的一两百毫秒被藏住；
        仍然走 call()，复用缓存、熔断和降级。阈值过滤由门控按 passes_threshold 做。
        """
        want = max(top_k, self.RERANK_CANDIDATES) if self._reranker is not None else top_k
        result = await self.call(tool_name, {"query": query, "top_k": want, "domain": domain}, use_cache=True)
        if not result.success or not isinstance(result.data, list):
            return []
        items = [item for item in result.data if isinstance(item, dict) and not item.get("fallback")]
        ranked, _ = await self._rerank(query, items)
        return ranked[:top_k]

    # ── 结果重排（解决召回不好）──────────────────────────────────────────────

    async def _rerank(self, query: str, items: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], bool]:
        """有重排器就按 cross-encoder 分数排序；没有或失败时按原来的混合分排序。返回 (结果, 是否重排)。"""
        if self._reranker is not None and items:
            ranked = await self._reranker.rerank(query, items[: self.RERANK_CANDIDATES])
            if ranked is not None:
                return ranked, True
        return sorted(items, key=lambda item: float(item.get("score", 0) or 0), reverse=True), False

    # ── 缓存 ──────────────────────────────────────────────────────────────────

    def invalidate_cache(self, tool_name: Optional[str] = None) -> int:
        """清空工具缓存（默认全部，传 tool_name 只清该工具）。返回清掉的条数。

        知识库写入后调用，保证新文档立即可检索，而不是等 TTL 过期。
        """
        if tool_name is None:
            removed = len(self._cache)
            self._cache.clear()
        else:
            prefix = f"{tool_name}:"
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                del self._cache[k]
            removed = len(keys)
        if removed:
            logger.info(f"工具缓存已失效: tool={tool_name or '*'} removed={removed}")
        return removed

    def _cache_key(self, name: str, params: Dict) -> str:
        return f"{name}:{hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()}"

    def _get_cache(self, name: str, params: Dict) -> Optional[Any]:
        key = self._cache_key(name, params)
        if key in self._cache:
            data, expire_at = self._cache[key]
            if time.monotonic() < expire_at:
                return data
            del self._cache[key]
        return None

    def _set_cache(
        self,
        name: str,
        params: Dict,
        data: Any,
        ttl: float,
    ) -> None:
        if len(self._cache) >= 5000:
            # 清掉最旧的 1/4
            for k in list(self._cache)[:1250]:
                del self._cache[k]
        self._cache[self._cache_key(name, params)] = (data, time.monotonic() + ttl)

    # ── 参数校验 ──────────────────────────────────────────────────────────────

    _TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
        """根据工具的 JSON Schema 校验参数，不合法时抛出 ValueError。"""
        schema = tool.schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for name in required:
            if name not in params:
                raise ValueError(f"工具 {tool.name} 缺少必需参数: {name}")

        for key, value in params.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and expected_type in self._TYPE_MAP:
                    if not isinstance(value, self._TYPE_MAP[expected_type]):
                        raise ValueError(
                            f"工具 {tool.name} 参数 {key} 类型错误: 期望 {expected_type}，实际 {type(value).__name__}"
                        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        return safe_text(value)

    # ── 统计 ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": t.stats.total,
                "success_rate": round(t.stats.success_rate, 3),
                "avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "consecutive_fails": t.stats.consecutive_fails,
                "circuit_state": t.breaker.state.value,
            }
            for name, t in self._tools.items()
        }

