"""只读 MCP 入口：用内存里的客户端走一遍真实的 MCP 协议。没有装 mcp SDK 时整个文件跳过（后端镜像不含它）。"""
import asyncio
import json

import pytest

pytest.importorskip("mcp.server.fastmcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from mcp_server import EXPOSED, build_server  # noqa: E402


def _with_client(coro_fn):
    async def run():
        server = build_server()
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            return await coro_fn(client)
    return asyncio.run(run())


def _payload(result):
    return json.loads(result.content[0].text)


def test_only_read_only_tools_are_exposed_with_full_schemas():
    tools = _with_client(lambda c: c.list_tools()).tools
    assert {t.name for t in tools} == set(EXPOSED)
    assert "create_consultation_lead" not in {t.name for t in tools}
    assert all(t.annotations and t.annotations.readOnlyHint for t in tools)
    quote = next(t for t in tools if t.name == "quote_service_bundle")
    schema = json.dumps(quote.inputSchema)
    assert "items" in quote.inputSchema["properties"]
    assert '"maximum": 20' in schema and '"minItems": 1' in schema      # 约束和内部一致


def test_quote_matches_the_internal_calculation():
    from business.pricing import quote_bundle

    expected = quote_bundle([{"sku": "selection_full", "qty": 1}], referral=True)["total"]
    result = _with_client(lambda c: c.call_tool(
        "quote_service_bundle", {"items": [{"sku": "selection_full"}], "referral": True}))
    assert not result.isError
    assert _payload(result)["total"] == expected


def test_invalid_arguments_are_rejected_by_the_protocol_layer():
    result = _with_client(lambda c: c.call_tool(
        "quote_service_bundle", {"items": [{"sku": "selection_full", "qty": 99}]}))
    assert result.isError


def test_business_failures_are_returned_not_raised():
    result = _with_client(lambda c: c.call_tool(
        "quote_service_bundle", {"items": [{"sku": "no_such_service"}]}))
    assert not result.isError
    assert _payload(result)["success"] is False
