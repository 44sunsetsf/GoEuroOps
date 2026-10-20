"""导出 retrieval_v1 每条问题经 RAG 门控实际注入的片段，供 knowledge_compare 当 RAG 对照组。

线上的向量模型和重排模型走 API，key 只在服务器上，所以在线上容器里跑（只建临时知识库，不写线上数据）：
    docker cp evaluation/rag_prefetch_dump.py goeuroops-backend-1:/tmp/ &&
    docker exec -w /app goeuroops-backend-1 python /tmp/rag_prefetch_dump.py > rag_prefetch_retrieval_v1.json
"""
import asyncio
import json
import sys
import tempfile
import time
sys.path.insert(0, "/app")
from core.rag_gate import RagGate, retrieval_query
from evaluation.retrieval_benchmark import build_manager, load_dataset, history_of, prefetch

async def main():
    rows = load_dataset("retrieval_v1")
    out = {}
    with tempfile.TemporaryDirectory(prefix="kb-dump-") as tmp:
        manager, kb = build_manager(tmp)
        async def search_fn(q, k, d):
            return await manager.search_fast("knowledge_search", q, top_k=k, domain=d)
        gate = RagGate(search_fn, enabled=True)
        for r in rows:
            q = retrieval_query(r["query"], history_of(r))
            t0 = time.monotonic()
            items = await prefetch(gate, r, q)
            out[r["id"]] = {"ms": round((time.monotonic() - t0) * 1000, 1),
                            "items": [{"title": i.get("title", ""), "content": i.get("content", ""),
                                       "score": i.get("score")} for i in items]}
        stats = kb.stats()
    json.dump({"embedding_model": stats["embedding_model"], "reranker": getattr(manager._reranker, "model_name", None),
               "chunks": stats["total_chunks"], "rows": out}, sys.stdout, ensure_ascii=False)

asyncio.run(main())
