"""
原话档案：用户说过的每句陈述原样存一条（ChromaDB），写进去马上就能被语义检索到。

只存原文和整理后补上的事件时间；向量按原文算。用来回答档案字段以外的细节（“我数学建模拿了什么奖”），
也作为档案每个字段的出处。保留 RETENTION_DAYS 天，过期的在整理时删掉；用户要求删除时全部删掉。
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

RETENTION_DAYS = 180
SHOW_CHARS = 300        # 每条原话放进提示词时最多多少字


def render(meta: Dict[str, Any]) -> str:
    said = (meta.get("ts") or "")[:10]
    when = f"[{meta['event_time']}｜说于 {said}]" if meta.get("event_time") else (f"[{said}]" if said else "")
    text = (meta.get("text") or "").strip()
    if len(text) > SHOW_CHARS:
        text = text[:SHOW_CHARS].rstrip() + "…"
    return f"{when} {text}".strip()


class UtteranceArchive:
    """ChromaDB 集合的薄封装，所有方法都是同步的，调用方放进线程里跑。"""

    def __init__(self, collection: Any, embedder: Any):
        self._col = collection
        self._embedder = embedder

    @staticmethod
    def note_id(user_id: str, text: str) -> str:
        return hashlib.md5(f"{user_id}\x00{text}".encode()).hexdigest()

    def add(self, user_id: str, conv_id: str, text: str, ts: datetime) -> None:
        """同一句话再说一遍不重复存，只刷新时间和会话，并标成待整理（改回以前的说法时，档案要按新的时间更新）。"""
        nid = self.note_id(user_id, text)
        got = self._col.get(ids=[nid])
        stamp = {"conv_id": conv_id, "ts": ts.isoformat(), "ts_num": ts.timestamp()}
        if got.get("ids"):
            self._col.update(ids=[nid], metadatas=[{**(got.get("metadatas") or [{}])[0], **stamp,
                                                    "consolidated": "0", "tries": 0}])
            return
        self._col.add(ids=[nid], documents=[text], metadatas=[{
            "user_id": user_id, "text": text, "consolidated": "0", "tries": 0, "event_time": "", **stamp}])

    def pending(self, user_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        got = self._col.get(where={"$and": [{"user_id": user_id}, {"consolidated": "0"}]})
        rows = list(zip(got.get("ids") or [], [m or {} for m in got.get("metadatas") or []]))
        return sorted(rows, key=lambda r: r[1].get("ts", ""))

    def update(self, nid: str, meta: Dict[str, Any]) -> None:
        self._col.update(ids=[nid], metadatas=[meta])

    def nearest(self, text: str, where: Dict[str, Any], k: int) -> List[Tuple[str, Dict[str, Any]]]:
        """按向量取最近的 k 条。

        ChromaDB 0.5 的 HNSW 带过滤条件查询时偶尔会报“Cannot return the results in a contigious 2D array”，
        这时在这位用户的原话里精确计算距离：单个用户最多几百条，精确算也很快。
        """
        qvec = self._embed_query(text)
        try:
            res = self._col.query(query_embeddings=[qvec], n_results=k, where=where, include=["metadatas"])
            return list(zip((res.get("ids") or [[]])[0], [m or {} for m in (res.get("metadatas") or [[]])[0]]))
        except Exception as ex:                     # noqa: BLE001
            if "contig" not in str(ex):
                raise
            got = self._col.get(where=where, include=["embeddings", "metadatas"])
            embs = got.get("embeddings")            # numpy 数组，不能写 `or []`
            embs = [] if embs is None else list(embs)
            rows = list(zip(got.get("ids") or [], embs, got.get("metadatas") or []))
            rows.sort(key=lambda r: -float(sum(a * b for a, b in zip(qvec, r[1]))))
            return [(i, m or {}) for i, _, m in rows[:k]]

    def _embed_query(self, text: str) -> List[float]:
        if hasattr(self._embedder, "embed_query"):      # bge 系列：查询带检索指令编码
            return list(self._embedder.embed_query(text))
        return [float(x) for x in self._embedder([text])[0]]

    def purge_older_than(self, user_id: str, cutoff: datetime) -> None:
        self._col.delete(where={"$and": [{"user_id": user_id}, {"ts_num": {"$lt": cutoff.timestamp()}}]})

    def delete_user(self, user_id: str) -> None:
        self._col.delete(where={"user_id": user_id})
