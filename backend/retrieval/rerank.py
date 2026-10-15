"""检索结果重排：调用 bge-reranker（cross-encoder）的远端 API，给“问题-片段”打 0–1 的相关度。

和向量 API 共用密钥与地址（硅基流动 OpenAI 兼容接口的 /rerank）。没有配置密钥时不启用，
检索按原来的混合分排序。接口失败、超时或熔断时返回 None，调用方退回混合分，不影响回答。

环境变量：
  GOEUROOPS_RERANK            off 关闭（默认开启，有密钥才生效）
  GOEUROOPS_RERANK_MODEL      默认 BAAI/bge-reranker-v2-m3
  GOEUROOPS_RERANK_MIN_SCORE  重排分数低于它的片段不注入、不返回（默认见 RERANK_MIN_SCORE）
  GOEUROOPS_RERANK_TIMEOUT    秒，默认 1.5
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from core.config import env_float
from retrieval.embeddings import DEFAULT_API_BASE_URL
from tooling.breaker import CircuitBreaker

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_MIN_SCORE = env_float("GOEUROOPS_RERANK_MIN_SCORE", 0.1)
HYBRID_MIN_SCORE = env_float("GOEUROOPS_RAG_PREFETCH_MIN_SCORE", 0.4)   # 没有重排时，混合分（向量 0.75 + 字面 0.25）的下限


def passes_threshold(item: Dict[str, Any], rerank_min: Optional[float] = None, hybrid_min: Optional[float] = None) -> bool:
    """重排过的看重排分数，没重排的看混合分：两种分数的刻度不同，阈值也不同。"""
    if item.get("reranked"):
        return float(item.get("score", 0) or 0) >= (RERANK_MIN_SCORE if rerank_min is None else rerank_min)
    return float(item.get("score", 0) or 0) >= (HYBRID_MIN_SCORE if hybrid_min is None else hybrid_min)


class ApiReranker:
    def __init__(self, model: str, api_key: str, base_url: str = DEFAULT_API_BASE_URL, timeout_s: float = 1.5):
        self.model_name = model
        self._url = base_url.rstrip("/") + "/rerank"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout_s
        self._breaker = CircuitBreaker(failure_threshold=3, recovery_s=60.0)

    async def scores(self, query: str, documents: List[str]) -> Optional[List[float]]:
        """返回和 documents 一一对应的相关度；失败、超时或熔断中返回 None。"""
        if not documents:
            return []
        ticket = self._breaker.acquire()
        if ticket is None:
            return None
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url, headers=self._headers, json={
                    "model": self.model_name, "query": query, "documents": documents,
                    "top_n": len(documents), "return_documents": False,
                })
                resp.raise_for_status()
                out = [0.0] * len(documents)
                for row in resp.json().get("results") or []:
                    out[int(row["index"])] = float(row["relevance_score"])
        except Exception as ex:                     # noqa: BLE001 —— 重排是增强，失败就退回混合分
            self._breaker.record_failure(ticket)
            logger.warning("重排接口失败，退回混合分排序: %s", ex)
            return None
        self._breaker.record_success(ticket)
        return out

    async def rerank(self, query: str, items: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """按重排分数从高到低返回新列表；score 换成重排分数，原来的混合分留在 hybrid_score。"""
        scores = await self.scores(query, [rerank_text(item) for item in items])
        if scores is None:
            return None
        ranked = [{**item, "hybrid_score": item.get("score", 0.0), "score": round(s, 4), "reranked": True}
                  for item, s in zip(items, scores)]
        return sorted(ranked, key=lambda item: item["score"], reverse=True)


def rerank_text(item: Dict[str, Any]) -> str:
    """给重排模型看的文本：标题 + 片段。"""
    return f"{item.get('title', '')}\n{item.get('content', '')}".strip()


def load_reranker() -> Optional[ApiReranker]:
    if os.getenv("GOEUROOPS_RERANK", "on").strip().lower() in {"0", "off", "false", "no"}:
        return None
    key = os.getenv("GOEUROOPS_RERANK_API_KEY", "").strip() or os.getenv("GOEUROOPS_EMBEDDING_API_KEY", "").strip()
    if not key:
        logger.info("没有配置重排 API 密钥，检索按混合分排序")
        return None
    base = (os.getenv("GOEUROOPS_RERANK_BASE_URL", "").strip() or os.getenv("GOEUROOPS_EMBEDDING_BASE_URL", "").strip()
            or DEFAULT_API_BASE_URL)
    model = os.getenv("GOEUROOPS_RERANK_MODEL", "").strip() or DEFAULT_RERANK_MODEL
    logger.info("检索重排模型（远端 API）: %s", model)
    return ApiReranker(model, key, base, timeout_s=env_float("GOEUROOPS_RERANK_TIMEOUT", 1.5))
