"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与检索工具管理器集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（A-Mem 笔记）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import logging
from typing import Any, Dict, List, Optional

import chromadb

from core.text_embedding import ngram_profile
from retrieval.embeddings import get_shared_embedding_function

logger = logging.getLogger(__name__)

# 混合检索：向量相似度为主，字面覆盖度为辅。
# 中文短问句里常夹着 APS、uni-assist、TUM 这类专有名词，纯向量容易漏掉，
# 字面覆盖度能把真正提到这些词的片段拉上来。
VECTOR_WEIGHT = 0.75
LEXICAL_WEIGHT = 0.25


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    向量模型默认使用 bge-small-zh-v1.5（见 retrieval/embeddings.py），在客户端编码后写入
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
        # 优先连接独立 ChromaDB 服务；chroma_host 为空时直接用进程内嵌入式（小内存部署省掉一个容器）
        self._use_server = False
        try:
            if not chroma_host:
                raise ConnectionError("未配置 CHROMA_HOST")
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
        self._backfill_doc_ids()

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
        批量导入文档到知识库，返回新写入的片段数。

        documents 格式: [{"title": "...", "content": "...", "domain": "...", "doc_id": "..."}, ...]
        长文档会自动切片（每片 500 字）。

        domain 标注文档归属哪个 Agent（如 "consulting"/"billing"/"general"），
        检索时按调用方 Agent 的 domain 过滤，避免不同业务线的知识互相串场；
        不填时默认为 "shared"，对所有 Agent 可见。

        每篇文档有稳定的 doc_id（不传时由来源 + 标题生成）和内容版本 doc_version：
        同一篇文档再导入时，内容没变就跳过；变了就先写新版本，写成功后再删旧版本，
        库里任何时刻都至少有一个完整版本，也不会新旧并存。

        每个片段前会加上【标题】再做向量化：切片后的正文经常缺少主题词，
        带上标题能明显改善中文短片段的召回。
        """
        written = sum(self._upsert_document(doc, source=source, seed_version=seed_version) for doc in documents)
        if written:
            logger.info(f"知识库导入 {written} 个文档片段")
        return written

    def _upsert_document(self, doc: Dict[str, str], *, source: str, seed_version: Optional[str]) -> int:
        title   = doc.get("title", "")
        content = doc.get("content", "")
        domain  = str(doc.get("domain") or "shared")
        doc_id  = str(doc.get("doc_id") or "").strip() or document_id(source, title)
        version = document_version(title, domain, content)

        existing = self._collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        old = [(i, m or {}) for i, m in zip(existing.get("ids") or [], existing.get("metadatas") or [])]
        current = [(i, m) for i, m in old if m.get("doc_version") == version]
        if current:
            stale = [i for i, m in old if m.get("doc_version") != version]
            if stale:                                   # 上次换版本时删旧块失败，这次补删
                self._collection.delete(ids=stale)
            if seed_version and any(m.get("seed_version") != seed_version for _, m in current):
                self._collection.update(ids=[i for i, _ in current],
                                        metadatas=[{**m, "seed_version": seed_version} for _, m in current])
            return 0

        chunks = self._chunk_text(content, chunk_size=500)
        if chunks:
            meta = {"title": title, "total_chunks": len(chunks), "domain": domain, "source": source,
                    "doc_id": doc_id, "doc_version": version}
            if seed_version:
                meta["seed_version"] = seed_version
            # ChromaDB 会自动生成 Embedding；先写新版本，失败就抛出，旧版本原样保留
            self._collection.upsert(
                ids=[hashlib.md5(f"{doc_id}:{version}:{i}".encode()).hexdigest() for i in range(len(chunks))],
                documents=[f"【{title}】{chunk}" if title else chunk for chunk in chunks],
                metadatas=[{**meta, "chunk_index": i} for i in range(len(chunks))],
            )
        if old:
            self._collection.delete(ids=[i for i, _ in old])
        return len(chunks)

    def delete_document(self, doc_id: str) -> int:
        """删除一篇文档的所有片段，返回删除的片段数。种子文档由业务目录生成，不能在这里删。"""
        got = self._collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        ids = got.get("ids") or []
        if any((m or {}).get("source") == "seed" for m in got.get("metadatas") or []):
            raise ValueError("种子文档由业务目录生成，请修改 business/ 下的 YAML 或文章后重启")
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def list_documents(self) -> List[Dict[str, Any]]:
        """按文档汇总：标题、领域、来源、版本和片段数。同一 doc_id 出现多个版本说明有残留。"""
        data = self._collection.get(include=["metadatas"])
        docs: Dict[str, Dict[str, Any]] = {}
        for meta in data.get("metadatas") or []:
            meta = meta or {}
            doc = docs.setdefault(meta.get("doc_id", ""), {
                "doc_id": meta.get("doc_id", ""), "title": meta.get("title", ""), "domain": meta.get("domain", "shared"),
                "source": meta.get("source", "upload"), "versions": set(), "chunks": 0,
            })
            doc["versions"].add(meta.get("doc_version", ""))
            doc["chunks"] += 1
        return sorted(({**d, "versions": sorted(d["versions"])} for d in docs.values()),
                      key=lambda d: (d["source"], d["title"]))

    def uploaded_documents(self) -> List[Dict[str, str]]:
        """后台上传的文档（非种子）按片段顺序拼回整篇，供 LLM Wiki 当作额外页面。"""
        data = self._collection.get(where={"source": {"$ne": "seed"}}, include=["documents", "metadatas"])
        parts: Dict[str, List[tuple]] = {}
        meta_of: Dict[str, Dict[str, Any]] = {}
        for text, meta in zip(data.get("documents") or [], data.get("metadatas") or []):
            meta = meta or {}
            doc_id = meta.get("doc_id", "")
            prefix = f"【{meta.get('title', '')}】"
            body = text[len(prefix):] if meta.get("title") and text.startswith(prefix) else text
            parts.setdefault(doc_id, []).append((int(meta.get("chunk_index", 0)), body))
            meta_of[doc_id] = meta
        return [{"doc_id": d, "title": meta_of[d].get("title", ""), "domain": meta_of[d].get("domain", "shared"),
                 "content": "".join(b for _, b in sorted(chunks))} for d, chunks in sorted(parts.items())]

    async def list_documents_async(self) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.list_documents)

    async def delete_document_async(self, doc_id: str) -> int:
        return await asyncio.to_thread(self.delete_document, doc_id)

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
                    "doc_id":   meta.get("doc_id", title),
                    "title":    title,
                    "content":  doc[len(prefix):] if title and doc.startswith(prefix) else doc,
                    "score":    round(VECTOR_WEIGHT * vector_score + LEXICAL_WEIGHT * lexical, 4),
                    "vector_score": round(vector_score, 4),
                    "lexical_score": round(lexical, 4),
                    "chunk":    meta.get("chunk_index", 0),
                    "domain":   meta.get("domain", "shared"),
                })

        items.sort(key=lambda item: item["score"], reverse=True)
        # 同一篇文档的多个块只保留得分最高的一个，不重复占 Top-K
        best: Dict[str, Dict[str, Any]] = {}
        for item in items:
            best.setdefault(item["doc_id"], item)
        return list(best.values())[:top_k]

    async def search_async(self, query: str, top_k: int = 5, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.search, query, top_k, domain)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    # ── 检索工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为检索工具的 handler 注册。

        RetrievalManager.register(Tool(
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
        把种子文档同步到业务目录的当前内容。

        种子由 business/ 下的业务目录生成（见 business/kb_seed.py）。逐篇比对 doc_version：
        没变的跳过，变了的先写新版再删旧版；全部写成功后，才删掉业务目录里已经没有的种子文档
        和早期版本遗留的示例文档。中途失败（比如向量 API 不可用）就抛出，旧种子原样保留。
        用户自己导入的文档不受影响。
        """
        from business.kb_seed import build_seed_documents

        version, docs = build_seed_documents()
        expected = {document_id("seed", d["title"]): document_version(d["title"], str(d.get("domain") or "shared"), d["content"])
                    for d in docs}
        seeds = self._collection.get(where={"source": "seed"}, include=["metadatas"])
        present = {((m or {}).get("doc_id"), (m or {}).get("doc_version")) for m in seeds.get("metadatas") or []}
        if present == set(expected.items()):
            logger.info("知识库种子已是最新版本: %s", version)
            return {"seed_version": version, "reseeded": False}

        added = sum(self._upsert_document({**d, "doc_id": document_id("seed", d["title"])}, source="seed",
                                          seed_version=version) for d in docs)

        # 只保留和当前种子完全一致的片段；旧示例里也有“退款政策”这种和现有种子同名的标题，不能按标题删
        current = set(expected.items())
        old = self._collection.get(where={"$or": [{"source": "seed"}, {"title": {"$in": LEGACY_SEED_TITLES}}]},
                                   include=["metadatas"])
        stale = [i for i, m in zip(old.get("ids") or [], old.get("metadatas") or [])
                 if ((m or {}).get("doc_id"), (m or {}).get("doc_version")) not in current]
        if stale:
            self._collection.delete(ids=stale)
        logger.info("知识库种子已更新: version=%s docs=%d 新写入片段=%d 删除片段=%d", version, len(docs), added, len(stale))
        return {"seed_version": version, "reseeded": True, "documents": len(docs), "chunks": added, "removed": len(stale)}

    def _backfill_doc_ids(self) -> None:
        """早期写入的片段没有 doc_id：按来源 + 标题补上（只改元数据，不重算向量），之后才能按文档替换和删除。"""
        try:
            data = self._collection.get(include=["metadatas"])
            todo = [(i, m or {}) for i, m in zip(data.get("ids") or [], data.get("metadatas") or [])
                    if not (m or {}).get("doc_id")]
            if todo:
                self._collection.update(
                    ids=[i for i, _ in todo],
                    metadatas=[{**m, "doc_id": document_id(m.get("source", "upload"), m.get("title", ""))} for _, m in todo],
                )
                logger.info("已给 %d 个旧片段补上 doc_id", len(todo))
        except Exception as ex:                     # noqa: BLE001 —— 迁移失败不影响检索
            logger.warning("补 doc_id 失败: %s", ex)

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


def document_id(source: str, title: str) -> str:
    """没有显式 doc_id 时，同一来源下同一标题就是同一篇文档。"""
    return hashlib.md5(f"{source}\x1f{title}".encode()).hexdigest()


def document_version(title: str, domain: str, content: str) -> str:
    return hashlib.sha256(f"{title}\x1f{domain}\x1f{content}".encode()).hexdigest()[:12]


def _lexical_coverage(query: str, document: str) -> float:
    """查询中的 2/3-gram 和英文词有多大比例出现在文档里（0–1）。"""
    grams = ngram_profile(query)
    if not grams:
        return 0.0
    doc = (document or "").lower()
    hit = sum(1 for gram in grams if gram in doc)
    return hit / len(grams)
