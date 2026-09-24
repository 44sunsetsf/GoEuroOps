"""工具缓存：知识库写入后立即失效，而不是等 TTL。"""
import asyncio

from mcp.tool_manager import MCPToolManager, Tool


def make_manager(docs):
    async def search(params, context):
        return [d for d in docs if params["query"] in d]

    manager = MCPToolManager(api_key="test")
    manager.register(Tool(
        name="knowledge_search",
        description="test",
        handler=search,
        schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        cache_ttl=300,
    ))
    return manager


def test_cached_result_is_stale_until_invalidated():
    async def run():
        docs = ["瑞典申请截止 1 月中"]
        manager = make_manager(docs)
        first = await manager.call("knowledge_search", {"query": "瑞典"})
        docs.append("瑞典奖学金 2 月截止")

        stale = await manager.call("knowledge_search", {"query": "瑞典"})
        assert stale.cached and len(stale.data) == 1

        assert manager.invalidate_cache("knowledge_search") == 1
        fresh = await manager.call("knowledge_search", {"query": "瑞典"})
        assert not fresh.cached and len(fresh.data) == 2
        assert len(first.data) == 1

    asyncio.run(run())


def test_invalidate_only_targets_named_tool():
    async def run():
        manager = make_manager(["a"])
        manager._set_cache("other_tool", {"q": 1}, ["x"], 300)
        await manager.call("knowledge_search", {"query": "a"})
        assert manager.invalidate_cache("knowledge_search") == 1
        assert manager._get_cache("other_tool", {"q": 1}) is not None

    asyncio.run(run())
