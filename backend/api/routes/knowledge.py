"""知识库检索、写入、上传和统计接口。"""
import logging
from typing import Optional


from fastapi import File, HTTPException, Request, UploadFile

from api.demo_guard import (
    is_guest,
)

from fastapi import APIRouter
from api.schemas import BatchDocInput
from api.routes.chat import _check_quota
from api.state import services

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search")
async def search(request: Request, query: str, top_k: int = 5, domain: Optional[str] = None):
    """
    演示检索优化链路：查询改写 → 并行召回 → 去重 → 相关性重排/过滤 → Top-K。
    展示检索工具治理（改写、重排、缓存）的效果。domain 可选，用于验证按 Agent 领域隔离检索的效果。
    """
    if services.tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    if is_guest(request.headers):
        await _check_quota()   # 查询改写要调用模型，访客的调用计入每日额度
    result = await services.tool_manager.search_with_rewrite("knowledge_search", query, top_k=top_k, domain=domain)
    return {"query": query, "results": result.data, "reranked": result.reranked, "success": result.success, "error": result.error}


@router.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    tool = services.tool_manager._tools.get("knowledge_search") if services.tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = await kb.add_documents_async([
        {"title": d.title, "content": d.content, "domain": d.domain} for d in body.documents
    ])
    # 知识库变了，旧的检索缓存可能不含新文档，立即失效而不是等 TTL
    services.tool_manager.invalidate_cache("knowledge_search")
    total = await kb.doc_count_async()
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": total}


@router.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...), domain: Optional[str] = None):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "...", "domain": "..."}, ...]`

    domain 参数（可选）：归属 Agent 领域（consulting/billing/general 等），
    没有在文档自己的字段里指定 domain 时会用这个值兜底；都不填则所有 Agent 可见。

    文件大小限制：10MB
    """
    tool = services.tool_manager._tools.get("knowledge_search") if services.tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    if domain:
        for doc in docs:
            if isinstance(doc, dict):
                doc.setdefault("domain", domain)

    count = await kb.add_documents_async(docs)
    services.tool_manager.invalidate_cache("knowledge_search")
    total = await kb.doc_count_async()
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": total,
    }


@router.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = services.tool_manager._tools.get("knowledge_search") if services.tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return await kb.stats_async()
