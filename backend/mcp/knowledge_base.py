"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional

import chromadb

from core.text_embedding import ngram_profile
from mcp.embeddings import get_shared_embedding_function

logger = logging.getLogger(__name__)

# 混合检索：向量相似度为主，字面覆盖度为辅。
# 中文短问句里常夹着 APS、uni-assist、TUM 这类专有名词，纯向量容易漏掉，
# 字面覆盖度能把真正提到这些词的片段拉上来。
VECTOR_WEIGHT = 0.75
LEXICAL_WEIGHT = 0.25


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    向量模型默认使用 bge-small-zh-v1.5（见 mcp/embeddings.py），在客户端编码后写入
    ChromaDB；检索时向量召回 + 字面覆盖度混合打分。模型不可用时退回 ChromaDB
    默认的 all-MiniLM-L6-v2。
    """

    LEGACY_COLLECTION = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 不同向量模型的向量维度不同，不能混在一个 collection 里：按模型名区分 collection。
        self._embedder = get_shared_embedding_function()
        if self._embedder is not None:
            self.collection_name = f"kb_{self._embedder.collection_suffix}"[:60]
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedder,
                metadata={
                    "description": "GoEuroOps RAG 知识库",
                    "embedding_model": self._embedder.model_name,
                    "hnsw:space": "cosine",
                },
            )
            self._migrate_legacy_collection()
        else:
            self.collection_name = self.LEGACY_COLLECTION
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "GoEuroOps RAG 知识库"},
            )

        # 按版本同步业务目录生成的种子文档（改了 catalog.yaml 重启即生效）
        try:
            self.sync_seed_documents()
        except Exception as ex:
            logger.error("知识库种子同步失败（不影响已有文档检索）: %s", ex)

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[Dict[str, str]],
        *,
        source: str = "upload",
        seed_version: Optional[str] = None,
    ) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "...", "domain": "..."}, ...]
        长文档会自动切片（每片 500 字）。

        domain 标注文档归属哪个 Agent（如 "consulting"/"billing"/"general"），
        检索时按调用方 Agent 的 domain 过滤，避免不同业务线的知识互相串场；
        不填时默认为 "shared"，对所有 Agent 可见。

        每个片段前会加上【标题】再做向量化：切片后的正文经常缺少主题词，
        带上标题能明显改善中文短片段的召回。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            domain  = str(doc.get("domain") or "shared")
            chunks  = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{source}_{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(f"【{title}】{chunk}" if title else chunk)
                meta = {"title": title, "chunk_index": i, "total_chunks": len(chunks), "domain": domain, "source": source}
                if seed_version:
                    meta["seed_version"] = seed_version
                metas.append(meta)

        if ids:
            # ChromaDB 会自动生成 Embedding；upsert 保证重复导入同一文档不会报错
            self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库导入 {len(ids)} 个文档片段")

        return len(ids)

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """异步导入文档；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        ChromaDB 内部自动将 query 转为向量，与存储的文档向量做余弦相似度匹配。
        domain 非空时只在该领域 + "shared" 范围内检索，实现按 Agent 隔离知识库。
        """
        where = {"domain": {"$in": [domain, "shared"]}} if domain else None
        recall_k = max(top_k * 4, 20)
        query_kwargs: Dict[str, Any] = {"n_results": recall_k, "where": where}
        if self._embedder is not None:
            query_kwargs["query_embeddings"] = [self._embedder.embed_query(query)]
        else:
            query_kwargs["query_texts"] = [query]
        results = self._collection.query(**query_kwargs)

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                title = meta.get("title", "")
                prefix = f"【{title}】"
                vector_score = 1.0 - dist  # cosine 空间下即余弦相似度
                lexical = _lexical_coverage(query, doc)
                items.append({
                    "title":    title,
                    "content":  doc[len(prefix):] if title and doc.startswith(prefix) else doc,
                    "score":    round(VECTOR_WEIGHT * vector_score + LEXICAL_WEIGHT * lexical, 4),
                    "vector_score": round(vector_score, 4),
                    "lexical_score": round(lexical, 4),
                    "chunk":    meta.get("chunk_index", 0),
                    "domain":   meta.get("domain", "shared"),
                })

        items.sort(key=lambda item: item["score"], reverse=True)
        return items[:top_k]

    async def search_async(self, query: str, top_k: int = 5, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.search, query, top_k, domain)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        domain = params.get("domain")
        return await self.search_async(query, top_k=top_k, domain=domain)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    # ── 种子文档 ──────────────────────────────────────────────────────────────

    def sync_seed_documents(self) -> Dict[str, Any]:
        """
        按 seed_version 同步种子文档。

        种子由 business/ 下的业务目录生成（见 business/kb_seed.py）。版本一致时
        什么都不做；不一致时删除旧种子（以及早期版本遗留的电商示例文档）后重新写入，
        用户自己导入的文档不受影响。
        """
        from business.kb_seed import build_seed_documents

        version, docs = build_seed_documents()
        current = self._collection.get(where={"seed_version": version}, limit=1)
        if current.get("ids"):
            logger.info("知识库种子已是最新版本: %s", version)
            return {"seed_version": version, "reseeded": False}

        self._collection.delete(where={"source": "seed"})
        legacy = self._collection.get(where={"title": {"$in": LEGACY_SEED_TITLES}})
        if legacy.get("ids"):
            self._collection.delete(ids=legacy["ids"])
            logger.info("已清理 %d 个旧版示例文档片段", len(legacy["ids"]))

        added = self.add_documents(docs, source="seed", seed_version=version)
        logger.info("知识库种子已更新: version=%s docs=%d chunks=%d", version, len(docs), added)
        return {"seed_version": version, "reseeded": True, "documents": len(docs), "chunks": added}

    def _migrate_legacy_collection(self) -> None:
        """
        从旧版 collection（默认英文向量模型）迁移用户自己导入的文档，然后删除旧 collection。
        种子文档和早期示例文档不迁移（会按新版本重新生成）。
        """
        try:
            legacy = self._client.get_collection(self.LEGACY_COLLECTION)
        except Exception:
            return
        data = legacy.get(include=["documents", "metadatas"])
        keep_ids, keep_docs, keep_metas = [], [], []
        for doc_id, doc, meta in zip(data.get("ids") or [], data.get("documents") or [], data.get("metadatas") or []):
            meta = meta or {}
            if meta.get("source") == "seed" or meta.get("title") in LEGACY_SEED_TITLES:
                continue
            keep_ids.append(doc_id)
            keep_docs.append(doc)
            keep_metas.append({**meta, "source": meta.get("source", "upload")})
        if keep_ids:
            self._collection.upsert(ids=keep_ids, documents=keep_docs, metadatas=keep_metas)
        self._client.delete_collection(self.LEGACY_COLLECTION)
        logger.info("已从旧知识库迁移 %d 个用户文档片段，并删除旧 collection", len(keep_ids))

    def stats(self) -> Dict[str, Any]:
        """片段总数、按 domain / 来源的分布，以及当前种子版本。"""
        data = self._collection.get(include=["metadatas"])
        by_domain: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        titles = set()
        seed_version = None
        for meta in data.get("metadatas") or []:
            meta = meta or {}
            by_domain[meta.get("domain", "shared")] = by_domain.get(meta.get("domain", "shared"), 0) + 1
            source = meta.get("source", "upload")
            by_source[source] = by_source.get(source, 0) + 1
            titles.add(meta.get("title", ""))
            seed_version = meta.get("seed_version") or seed_version
        return {
            "collection": self.collection_name,
            "embedding_model": self._embedder.model_name if self._embedder else "chromadb-default(all-MiniLM-L6-v2)",
            "total_chunks": len(data.get("ids") or []),
            "documents": len(titles),
            "by_domain": by_domain,
            "by_source": by_source,
            "seed_version": seed_version,
        }

    async def stats_async(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self.stats)


# 早期版本（通用电商客服示例 + 第一版留学占位内容）写入的种子标题，升级时一并清理
LEGACY_SEED_TITLES = [
    "退款政策", "订单查询", "账户安全", "会员与积分", "配送说明", "技术故障排查",
    "工作室服务介绍（示例参考信息）",
    "瑞典CS硕士申请概览（示例参考信息）",
    "德国CS硕士申请概览（示例参考信息）",
    "荷兰CS硕士申请概览（示例参考信息）",
    "芬兰CS硕士申请概览（示例参考信息）",
    "丹麦CS硕士申请概览（示例参考信息）",
    "申请材料与时间线概览（示例参考信息）",
]


def _lexical_coverage(query: str, document: str) -> float:
    """查询中的 2/3-gram 和英文词有多大比例出现在文档里（0–1）。"""
    grams = ngram_profile(query)
    if not grams:
        return 0.0
    doc = (document or "").lower()
    hit = sum(1 for gram in grams if gram in doc)
    return hit / len(grams)
