"""
A-Mem 魔改版记忆（GOEUROOPS_MEMORY_IMPL=amem）。

参考 A-Mem（arXiv 2502.12110，Zettelkasten 式笔记记忆），按本项目改造：
  - 工作记忆、画像、摘要式情景记忆全部沿用 MemoryManager，只把“原话”这一层换成笔记层（集合 amem_notes）；
    基类的逐句索引照常写入，切回 v2 不丢数据。
  - 用户的话**立刻**存成一条原文笔记（只算向量，不调模型）；攒够 NOTE_BATCH 句或会话压缩时，在后台**一次调用批量补全**：
    关键词、标签、背景、事件时间（把“去年夏天”按说话日期换算），以及和旧笔记的链接、取代关系。
    A-Mem 原版是每条消息同步调 2 次模型。
  - 演化**只追加版本**：旧笔记的背景被改写时，旧版本留在 versions 里，并重算向量（官方代码改了内容却不更新向量）。
    A-Mem 原版直接覆盖，改错了无法恢复，也回答不了“以前是什么”。
  - 检索：向量取前 NOTE_TOP_K 条 + 顺链接多取 LINK_EXPAND 条，带事件时间、背景和“后来有更新”的标记。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from core.llm_utils import NO_THINKING_KWARGS, extract_text_content
from memory.conversation_memory import MemoryManager, Message, MsgRole, _parse_json_object

logger = logging.getLogger(__name__)


class AgenticMemoryManager(MemoryManager):
    NOTE_BATCH      = 10    # 攒够多少句未补全的笔记就在后台补全一次
    NOTE_TOP_K      = 6     # 检索取几条笔记
    LINK_EXPAND     = 3     # 顺链接最多多取几条
    NEIGHBORS_EACH  = 3     # 补全时每句取几条已有笔记做候选
    NEIGHBOR_CAP    = 10    # 一批最多给模型看几条旧笔记
    MAX_TRIES       = 2     # 补全失败几次后放弃（原文笔记仍然可检索）

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._notes = self._chroma.get_or_create_collection("amem_notes")
        self._note_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._bg_tasks: set = set()

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(self, user_id: str, conv_id: str, role: MsgRole, content: str,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        if role == MsgRole.USER:
            msg = Message(role=role, content=self._safe_text(content))
            await self._add_raw_note(self._safe_text(user_id), self._safe_text(conv_id), msg)
        await super().add_message(user_id, conv_id, role, content, metadata)

    async def _add_raw_note(self, user_id: str, conv_id: str, msg: Message) -> None:
        """先存原文，马上就能被检索到；关键词、链接等等后台补全。"""
        text = msg.content.strip()
        if not self._worth_indexing(text):
            return
        ts = msg.timestamp.isoformat()
        try:
            await asyncio.wait_for(asyncio.to_thread(
                self._notes.add,
                ids=[hashlib.md5(f"{user_id}{conv_id}{ts}{text}".encode()).hexdigest()],
                documents=[text],
                metadatas=[{"user_id": user_id, "conv_id": conv_id, "ts": ts, "text": text, "enriched": "0",
                            "tries": 0, "keywords": "", "tags": "", "context": "", "event_time": "",
                            "links": "[]", "versions": "[]", "superseded_by": ""}],
            ), 10)
        except Exception as ex:                     # noqa: BLE001 —— 少一条笔记不能影响对话
            logger.warning(f"写入笔记失败: {ex}")

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        """对话接口每轮回复后在后台调用：先更新画像，再看要不要补全笔记。"""
        await super().update_profile(user_id, conv_id)
        try:
            await self.enrich_notes(self._safe_text(user_id), min_pending=self.NOTE_BATCH)
        except Exception as ex:                     # noqa: BLE001
            logger.warning(f"补全笔记失败: {ex}")

    async def _compress(self, user_id: str, conv_id: str, force: bool = False) -> None:
        await super()._compress(user_id, conv_id, force)
        # 压缩时把这位用户没补全的笔记都补上；放后台，不拖慢这一轮
        task = asyncio.create_task(self._enrich_quietly(self._safe_text(user_id)))
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _enrich_quietly(self, user_id: str) -> None:
        try:
            await self.enrich_notes(user_id, min_pending=1)
        except Exception as ex:                     # noqa: BLE001
            logger.warning(f"补全笔记失败: {ex}")

    async def wait_background(self) -> None:
        """等压缩触发的后台补全跑完（评测用：线上它们自己会跑完）。"""
        while self._bg_tasks:
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)

    async def flush_notes(self, user_id: str) -> int:
        """等后台任务结束并补全全部笔记（评测和测试用）。"""
        if self._bg_tasks:
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)
        return await self.enrich_notes(self._safe_text(user_id), min_pending=1)

    async def enrich_notes(self, user_id: str, min_pending: int = 1) -> int:
        """补全未补全的笔记，每 NOTE_BATCH 句一次模型调用。返回补全的条数。"""
        done = 0
        async with self._note_locks[user_id]:
            pending = await self._pending(user_id)
            if len(pending) < min_pending:
                return 0
            for i in range(0, len(pending), self.NOTE_BATCH):
                done += await self._enrich_batch(user_id, pending[i:i + self.NOTE_BATCH])
        return done

    async def _pending(self, user_id: str) -> List[Dict[str, Any]]:
        got = await asyncio.to_thread(self._notes.get, where={"$and": [{"user_id": user_id}, {"enriched": "0"}]})
        rows = [{"id": i, "meta": m or {}} for i, m in zip(got.get("ids") or [], got.get("metadatas") or [])]
        return sorted(rows, key=lambda r: r["meta"].get("ts", ""))

    async def _enrich_batch(self, user_id: str, batch: List[Dict[str, Any]]) -> int:
        batch_ids = {r["id"] for r in batch}
        neighbors: Dict[str, Dict[str, Any]] = {}
        for r in batch:
            if len(neighbors) >= self.NEIGHBOR_CAP:
                break
            res = await asyncio.to_thread(
                self._notes.query, query_texts=[r["meta"].get("text", "")], n_results=self.NEIGHBORS_EACH,
                where={"$and": [{"user_id": user_id}, {"enriched": "1"}]},
            )
            for nid, nmeta in zip((res.get("ids") or [[]])[0], (res.get("metadatas") or [[]])[0]):
                if nid not in batch_ids and nid not in neighbors and len(neighbors) < self.NEIGHBOR_CAP:
                    neighbors[nid] = nmeta or {}

        m_ids = {f"M{k + 1}": r for k, r in enumerate(batch)}
        n_ids = {f"N{k + 1}": (nid, meta) for k, (nid, meta) in enumerate(neighbors.items())}
        msgs = "\n".join(f"{k}（说于 {r['meta'].get('ts', '')[:10]}）：{r['meta'].get('text', '')}" for k, r in m_ids.items())
        olds = "\n".join(f"{k}（说于 {m.get('ts', '')[:10]}，背景：{m.get('context', '')}）：{m.get('text', '')}"
                         for k, (_, m) in n_ids.items()) or "（无）"
        prompt = self._safe_text(
            "你在整理一位用户的记忆笔记。下面 M 开头的是他新说的话，N 开头的是以前的笔记。\n"
            "对每条 M：给 3–6 个以后能用来搜到它的关键词（人名、地点、事件、物品，用原话的语言）、最多 3 个主题标签、"
            "一句话背景（这句话在讲什么）、事件时间（话里的事发生在什么时候，按说话日期换算成 YYYY-MM-DD 或 YYYY-MM；"
            "话里自带日期的以话里为准；说不出就留空）、和哪些 M/N 有关联（links）、是否取代了某条 N（同一件事后来改了主意或更新了，supersedes）。\n"
            "如果新信息让某条 N 的背景需要补充或更正，在 neighbor_updates 里给出新的背景（只改背景，不改原话）。\n"
            "只返回 JSON："
            '{"notes": [{"m": "M1", "keywords": [], "tags": [], "context": "", "event_time": "", "links": [], "supersedes": []}],'
            ' "neighbor_updates": [{"n": "N1", "context": ""}]}\n\n'
            f"新说的话：\n{msgs}\n\n以前的笔记：\n{olds}"
        )
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=2000, temperature=0.0,
                messages=[{"role": "user", "content": prompt}], **NO_THINKING_KWARGS,
            )
            obj = _parse_json_object(extract_text_content(resp.content))
            notes = {n.get("m"): n for n in obj.get("notes") or [] if isinstance(n, dict)}
        except Exception as ex:                     # noqa: BLE001 —— 原文笔记保留，下次再补；多次失败就放弃补全
            logger.warning(f"补全笔记的模型输出无效: {ex}")
            for r in batch:
                tries = int(r["meta"].get("tries") or 0) + 1
                meta = {**r["meta"], "tries": tries, "enriched": "1" if tries >= self.MAX_TRIES else "0"}
                await asyncio.to_thread(self._notes.update, ids=[r["id"]], metadatas=[meta])
            return 0

        def resolve(ref: str) -> Optional[str]:
            if ref in m_ids:
                return m_ids[ref]["id"]
            if ref in n_ids:
                return n_ids[ref][0]
            return None

        link_back: Dict[str, List[str]] = defaultdict(list)
        superseded: Dict[str, str] = {}
        for key, r in m_ids.items():
            n = notes.get(key) or {}
            meta = dict(r["meta"])
            kws = [str(k) for k in n.get("keywords") or []][:6]
            tags = [str(t) for t in n.get("tags") or []][:3]
            links = [x for x in (resolve(str(ref)) for ref in n.get("links") or []) if x and x != r["id"]]
            for ref in n.get("supersedes") or []:
                old = resolve(str(ref))
                if old and old != r["id"]:
                    superseded[old] = r["id"]
            for target in links:
                link_back[target].append(r["id"])
            meta.update(enriched="1", keywords="、".join(kws), tags="、".join(tags),
                        context=str(n.get("context") or "")[:200], event_time=str(n.get("event_time") or "")[:20],
                        links=json.dumps(sorted(set(links))))
            await asyncio.to_thread(self._notes.update, ids=[r["id"]], metadatas=[meta], documents=[self._note_doc(meta)])

        # 旧笔记：背景更新（只追加版本）、反向链接、取代标记
        updates = {u.get("n"): str(u.get("context") or "") for u in obj.get("neighbor_updates") or [] if isinstance(u, dict)}
        touched = set(link_back) | set(superseded) | {n_ids[k][0] for k in updates if k in n_ids}
        for nid in touched:
            got = await asyncio.to_thread(self._notes.get, ids=[nid])
            if not got.get("ids"):
                continue
            meta = dict((got.get("metadatas") or [{}])[0] or {})
            links = set(json.loads(meta.get("links") or "[]")) | set(link_back.get(nid, []))
            meta["links"] = json.dumps(sorted(links))
            if nid in superseded:
                meta["superseded_by"] = superseded[nid]
            key = next((k for k, (i, _) in n_ids.items() if i == nid), None)
            new_ctx = updates.get(key, "").strip() if key else ""
            if new_ctx and new_ctx != meta.get("context", ""):
                versions = json.loads(meta.get("versions") or "[]")
                versions.append({"context": meta.get("context", ""), "replaced_at": batch[-1]["meta"].get("ts", "")})
                meta["versions"] = json.dumps(versions, ensure_ascii=False)
                meta["context"] = new_ctx[:200]
            await asyncio.to_thread(self._notes.update, ids=[nid], metadatas=[meta], documents=[self._note_doc(meta)])
        return len(batch)

    @staticmethod
    def _note_doc(meta: Dict[str, Any]) -> str:
        """算向量用的文本：原话 + 背景 + 关键词 + 标签（A-Mem 的做法）。"""
        parts = [meta.get("text", "")]
        if meta.get("context"):
            parts.append(f"背景：{meta['context']}")
        if meta.get("keywords"):
            parts.append(f"关键词：{meta['keywords']}")
        if meta.get("tags"):
            parts.append(f"标签：{meta['tags']}")
        return " ".join(parts)

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def _search_turns(self, user_id: str, query: str, exclude: List[str]) -> List[str]:
        """替换基类“原话”层的检索：取笔记，顺链接扩展。get_context 不用改。"""
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        skip = {self._safe_text(t).strip() for t in exclude}
        user_id = self._safe_text(user_id)
        try:
            res = await asyncio.wait_for(asyncio.to_thread(
                self._notes.query, query_texts=[query_text],
                n_results=self.NOTE_TOP_K + len(skip), where={"user_id": user_id},
            ), 10)
            hits: List[tuple] = []
            for nid, meta in zip((res.get("ids") or [[]])[0], (res.get("metadatas") or [[]])[0]):
                meta = meta or {}
                if meta.get("text", "").strip() in skip:
                    continue
                hits.append((nid, meta))
                if len(hits) >= self.NOTE_TOP_K:
                    break
            have = {h[0] for h in hits}
            want: List[str] = []
            for _, meta in hits:
                for lid in json.loads(meta.get("links") or "[]"):
                    if lid not in have and lid not in want:
                        want.append(lid)
            want = want[: self.LINK_EXPAND]
            if want:
                got = await asyncio.to_thread(self._notes.get, ids=want)
                for nid, meta in zip(got.get("ids") or [], got.get("metadatas") or []):
                    if (meta or {}).get("text", "").strip() not in skip:
                        hits.append((nid, meta or {}))
            return self._dedupe_texts([self._render_note(m) for _, m in hits])
        except Exception as ex:                     # noqa: BLE001
            logger.warning(f"笔记检索失败: {ex}")
            return []

    def _render_note(self, meta: Dict[str, Any]) -> str:
        said = (meta.get("ts") or "")[:10]
        when = f"[{meta['event_time']}｜说于 {said}]" if meta.get("event_time") else (f"[{said}]" if said else "")
        text = meta.get("text", "").strip()
        if len(text) > self.TURN_SHOW_CHARS:
            text = text[: self.TURN_SHOW_CHARS].rstrip() + "…"
        out = f"{when} {text}".strip()
        if meta.get("context"):
            out += f"（背景：{meta['context']}）"
        if meta.get("superseded_by"):
            out += "（后来有更新，以新的为准）"
        return out
