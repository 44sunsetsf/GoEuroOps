"""
原话笔记（ChromaDB），按 A-Mem（arXiv 2502.12110，Zettelkasten 式笔记记忆）的思路组织：

  - 用户说的每句陈述立刻原样存成一条笔记，马上就能被检索到；
  - 整理时补上关键词、标签、一句话背景、事件时间，以及和其他笔记的链接、取代关系；
  - 演化只追加：新信息让旧笔记的背景需要更正时，旧背景留在 versions 里，并按新内容重算向量；
  - 向量按“原话 + 背景 + 关键词 + 标签”算，检索时顺着链接多取几条。
保留 RETENTION_DAYS 天，过期的在整理时删掉；用户要求删除时全部删掉。
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RETENTION_DAYS = 180
SHOW_CHARS = 300        # 每条笔记放进提示词时最多多少字


def note_doc(meta: Dict[str, Any]) -> str:
    """算向量用的文本：原话 + 背景 + 关键词 + 标签（A-Mem 的做法）。"""
    parts = [meta.get("text", "")]
    if meta.get("context"):
        parts.append(f"背景：{meta['context']}")
    if meta.get("keywords"):
        parts.append(f"关键词：{meta['keywords']}")
    if meta.get("tags"):
        parts.append(f"标签：{meta['tags']}")
    return " ".join(parts)


def render(meta: Dict[str, Any]) -> str:
    said = (meta.get("ts") or "")[:10]
    when = f"[{meta['event_time']}｜说于 {said}]" if meta.get("event_time") else (f"[{said}]" if said else "")
    text = (meta.get("text") or "").strip()
    if len(text) > SHOW_CHARS:
        text = text[:SHOW_CHARS].rstrip() + "…"
    out = f"{when} {text}".strip()
    if meta.get("context"):
        out += f"（背景：{meta['context']}）"
    if meta.get("superseded_by"):
        out += "（后来有更新，以新的为准）"
    return out


class NoteStore:
    """ChromaDB 集合的薄封装，所有方法都是同步的，调用方放进线程里跑。"""

    def __init__(self, collection: Any, embedder: Any):
        self._col = collection
        self._embedder = embedder

    @staticmethod
    def note_id(user_id: str, text: str) -> str:
        return hashlib.md5(f"{user_id}\x00{text}".encode()).hexdigest()

    def add(self, user_id: str, conv_id: str, text: str, ts: datetime) -> None:
        """同一句话再说一遍不重复存：刷新时间和会话，保留已有的补全，并标成待整理（改回以前的说法时，档案要按新的时间更新）。"""
        nid = self.note_id(user_id, text)
        got = self._col.get(ids=[nid])
        stamp = {"conv_id": conv_id, "ts": ts.isoformat(), "ts_num": ts.timestamp()}
        if got.get("ids"):
            self._col.update(ids=[nid], metadatas=[{**(got.get("metadatas") or [{}])[0], **stamp,
                                                    "consolidated": "0", "tries": 0}])
            return
        self._col.add(ids=[nid], documents=[text], metadatas=[{
            "user_id": user_id, "text": text, "consolidated": "0", "tries": 0, "event_time": "",
            "keywords": "", "tags": "", "context": "", "links": "[]", "versions": "[]", "superseded_by": "", **stamp}])

    def pending(self, user_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        got = self._col.get(where={"$and": [{"user_id": user_id}, {"consolidated": "0"}]})
        rows = list(zip(got.get("ids") or [], [m or {} for m in got.get("metadatas") or []]))
        return sorted(rows, key=lambda r: r[1].get("ts", ""))

    def get(self, ids: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
        got = self._col.get(ids=ids)
        return list(zip(got.get("ids") or [], [m or {} for m in got.get("metadatas") or []]))

    def update(self, nid: str, meta: Dict[str, Any], reembed: bool = False) -> None:
        """reembed：背景、关键词变了，按新内容重算向量。"""
        self._col.update(ids=[nid], metadatas=[meta], documents=[note_doc(meta)] if reembed else None)

    def nearest(self, text: str, where: Dict[str, Any], k: int) -> List[Tuple[str, Dict[str, Any]]]:
        """按向量取最近的 k 条。

        ChromaDB 0.5 的 HNSW 带过滤条件查询时偶尔会报“Cannot return the results in a contigious 2D array”，
        这时在这位用户的笔记里精确计算距离：单个用户最多几百条，精确算也很快。
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


def resolve(ref: str, m_ids: Dict[str, str], n_ids: Dict[str, str]) -> Optional[str]:
    """模型输出里的 M1 / N2 换成笔记 id。"""
    return m_ids.get(ref) or n_ids.get(ref)
