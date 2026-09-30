"""把只读的业务查询工具用 MCP 协议对外提供，给别的客户端用（例如顾问在 Claude Code 里直接查价目、算报价）。

内部的 Agent 和工具在同一个进程里，不需要 MCP；这里只是给外部客户端的入口：
  - 只暴露只读工具：服务与价目、国家申请资料、付款政策、报价。
  - 写入类工具（登记线索、转顾问）不暴露：外部客户端没有用户身份，不应该能写数据。
  - 和内部 Agent 共用同一份工具定义（Pydantic 入参、返回值校验、执行策略）和同一个网关，
    所以校验、超时、报价不变量检查在两边完全一致，不会出现两套逻辑。

运行（本地 stdio，不部署到服务器）：
    pip install -r requirements-mcp.txt
    python mcp_server.py
接入 Claude Code：
    claude mcp add goeuroops-studio -- python /绝对路径/backend/mcp_server.py
"""
from __future__ import annotations

import inspect
import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP                      # noqa: E402
from mcp.types import ToolAnnotations                       # noqa: E402

from agents.tools import AgentToolSpec, billing_tools, consulting_tools  # noqa: E402
from tooling.gateway import ToolGateway                     # noqa: E402

# 对外开放的工具：全部是只读查询或纯计算
EXPOSED = (
    "lookup_service_offering",
    "quote_service_bundle",
    "lookup_country_admissions_overview",
    "get_payment_policy",
)


def _readonly_specs() -> Dict[str, AgentToolSpec]:
    specs = {**consulting_tools(None), **billing_tools()}
    chosen = {name: specs[name] for name in EXPOSED}
    for spec in chosen.values():                            # 双保险：策略里不是只读的一律不开放
        if spec.policy.side_effect != "read":
            raise RuntimeError(f"{spec.name} 不是只读工具，不能通过 MCP 暴露")
    return chosen


def _make_function(spec: AgentToolSpec, gateway: ToolGateway):
    """按 Pydantic 入参模型生成函数签名，让 FastMCP 给客户端的参数说明、约束和默认值与内部完全一致。"""
    model = spec.input_model
    params = [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=field, annotation=field.annotation)
        for name, field in model.model_fields.items()
    ]

    async def tool_fn(**kwargs: Any) -> Dict[str, Any]:
        args = {k: v for k, v in kwargs.items() if v is not None}
        outcome = await gateway.call(spec, None, args)
        if not outcome.ok:                                   # 超时、异常、返回值没通过校验：作为 MCP 错误返回
            raise ValueError(outcome.error or "工具执行失败")
        return outcome.result                                # 业务层面的失败（success=false）照常返回

    tool_fn.__signature__ = inspect.Signature(params)        # type: ignore[attr-defined]
    tool_fn.__annotations__ = {p.name: p.annotation for p in params} | {"return": Dict[str, Any]}
    tool_fn.__name__ = spec.name
    return tool_fn


def build_server(gateway: Optional[ToolGateway] = None) -> FastMCP:
    gateway = gateway or ToolGateway()
    server = FastMCP("goeuroops-studio")
    for spec in _readonly_specs().values():
        server.add_tool(
            _make_function(spec, gateway),
            name=spec.name,
            description=spec.description,
            annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
        )
    return server


if __name__ == "__main__":
    build_server().run()
