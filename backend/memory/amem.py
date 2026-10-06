"""
GoEuroOps 的记忆模块，按 A-Mem（arXiv 2502.12110，Zettelkasten 式笔记记忆）的思路实现。

记忆单位是用户说过的原话，一句一条“笔记”：
  1. 最近对话（Redis）：当前会话最近 WINDOW 条原文，滑动窗口，不调模型。
  2. 笔记（ChromaDB）：用户说的每句陈述立刻存成一条原文笔记，马上就能被检索到；
     攒够一批、用户换了会话、或者用户停下来 IDLE_SECONDS 秒时，在后台**一次模型调用批量补全**：
     关键词、标签、一句话背景、事件时间（把“去年夏天”按说话日期换算）、关键事实字段（专业、预算……），
     以及和旧笔记的链接、取代关系。A-Mem 原版是每条消息同步调 2 次模型。
  3. 演化只追加：新信息让旧笔记的背景需要更正时，旧背景留在 versions 里，并按新内容重算向量；
     改了主意的旧笔记标记 superseded_by，回答“现在”和“以前”都有依据。
  4. 关键事实不单独维护画像：补全时给笔记标上字段和取值，取用时同一字段按说话时间排，以最新为准、旧值留档。

检索：笔记向量取前 NOTE_TOP_K 条，顺链接多取 LINK_EXPAND 条，按说话时间先后排。
向量模型和知识库共用一个（默认 bge-small-zh，中文检索）；ChromaDB 默认的英文模型对中文几乎没有区分度。
任何一步失败都按“少一点记忆”继续，不影响对话；补全失败的笔记保留原文，下次再补。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import redis.asyncio as redis

from core.config import DEFAULT_MODEL
from core.llm_utils import NO_THINKING_KWARGS, extract_text_content, make_client, safe_text

logger = logging.getLogger(__name__)

ASSISTANT_MSG_CHARS = 300   # 提示词里每条助手回复最多放多少字

# 关键事实的字段：补全时给笔记标上其中一个，取用时按字段汇总
FACT_SLOTS: Dict[str, str] = {
    "major": "本科专业",
    "school": "本科院校",
    "gpa": "GPA / 均分",
    "language": "语言考试成绩（写明考试名和分数，如 雅思 6.5）",
    "target_country": "目标国家",
    "target_city": "目标城市",
    "intake": "计划入学时间",
    "budget": "预算",
    "graduation": "毕业时间 / 工作年限",
    "experience": "实习、工作或项目经历",
    "other": "其他明确的偏好或约束",
}
# 这两个字段可以同时有好几条（几段经历、几个偏好），全部列出；其他字段只有一个当前值，后说的为准
LIST_SLOTS = {"experience", "other"}
LIST_SHOW = 5

_SELF_INFO = re.compile(
    r"\d|专业|本科|硕士|毕业|学校|大学|学院|gpa|绩点|均分|雅思|托福|ielts|toefl|gre|预算|万|入学|春季|秋季|冬季|学期|"
    r"实习|工作|经历|项目|决定|改成|改申|不去|打算|想去|计划|目标|"
    r"瑞典|德国|荷兰|芬兰|丹麦|挪威|英国|美国|北欧|欧洲|新加坡|香港|澳洲|加拿大",
    re.IGNORECASE,
)

_UNSET = object()


def has_self_info(text: str) -> bool:
    """这句用户消息是不是在讲自己的情况：带“我”，并且出现数字或专业、成绩、国家、预算这类词。"""
    t = text or ""
    return "我" in t and bool(_SELF_INFO.search(t))


def _parse_json_object(text: str) -> Dict[str, Any]:
    raw = (text or "").strip().strip("`")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"没有找到 JSON：{raw[:80]!r}")
    return json.loads(raw[start:end + 1])


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=datetime.now)
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """传给 Agent 的记忆。"""
    recent_messages: List[Message]                          # 当前会话最近的原文
    notes:           List[str] = field(default_factory=list)  # 和这次提问相关的笔记（按说话时间先后）
    facts:           List[str] = field(default_factory=list)  # 关键事实：每个字段的最新值和以前的说法

    def to_prompt_text(self) -> str:
        parts = []
        if self.facts:
            parts.append("[用户的关键事实]（用户本人说过的，后说的为准，括号里是以前的说法）\n"
                         + "\n".join(f"- {safe_text(f)}" for f in self.facts))
        if self.notes:
            parts.append("[用户以前说过的相关原话]（按说话时间先后）\n" + "\n".join(f"- {safe_text(n)}" for n in self.notes))
        if self.recent_messages:
            lines = ["[最近对话]"]
            for m in self.recent_messages:
                content = safe_text(m.content)
                if m.role == MsgRole.ASSISTANT and len(content) > ASSISTANT_MSG_CHARS:
                    content = content[:ASSISTANT_MSG_CHARS].rstrip() + "…（略）"
                lines.append(f"{m.role.value}: {content}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)


class MemoryManager:
    WINDOW          = 14     # 提示词里放最近多少条原文
    NOTE_BATCH      = 8      # 攒够多少句未补全的笔记就补全一次
    NOTE_TOP_K      = 6      # 检索取几条笔记
    LINK_EXPAND     = 3      # 顺链接最多多取几条
    NEIGHBORS_EACH  = 3      # 补全时每句取几条已有笔记做候选
    NEIGHBOR_CAP    = 10     # 一批最多给模型看几条旧笔记
    MAX_TRIES       = 2      # 补全失败几次后放弃（原文笔记仍然可检索）
    NOTE_MIN_CHARS  = 6      # 太短的话（“好的”“谢谢”）不存
    NOTE_SHOW_CHARS = 300    # 每条笔记放进提示词时最多多少字
    SEARCH_TIMEOUT  = 5.0    # 提问时检索笔记最多等多久（秒）
    IDLE_SECONDS    = 120    # 用户停下来这么久没说话，就把他没补全的笔记都补上（相当于会话结束）

    def __init__(
        self,
        redis_url:   str = "redis://localhost:6379/0",
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        api_key:     str = "",
        base_url:    Optional[str] = None,
        model:       str = DEFAULT_MODEL,
        embedder:    Any = _UNSET,
    ):
        self._client = make_client(api_key, base_url, "memory")
        self._model = model

        # 最近对话在每次对话的主链路上：Redis 慢或挂了不能拖住对话，连接和读写都设超时，失败按“没有记忆”继续
        self._redis_timeout = float(os.getenv("GOEUROOPS_MEMORY_REDIS_TIMEOUT", "1.0"))
        self._redis = redis.from_url(redis_url, decode_responses=True, socket_timeout=self._redis_timeout,
                                     socket_connect_timeout=self._redis_timeout)

        # ChromaDB：优先连独立服务（docker compose），连不上或 chroma_host 为空时用本地嵌入式
        try:
            if not chroma_host:
                raise ConnectionError("未配置 CHROMA_HOST")
            chroma = chromadb.HttpClient(host=chroma_host, port=chroma_port,
                                         settings=chromadb.Settings(anonymized_telemetry=False))
            chroma.heartbeat()
            logger.info(f"ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:                           # noqa: BLE001
            logger.info(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
            chroma = chromadb.PersistentClient(path=chroma_path, settings=chromadb.Settings(anonymized_telemetry=False))

        if embedder is _UNSET:
            from retrieval.embeddings import get_shared_embedding_function
            embedder = get_shared_embedding_function()
        if embedder is None:                        # 中文模型加载失败：退回 ChromaDB 默认模型，记忆照常可用
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            embedder, name = DefaultEmbeddingFunction(), "amem_notes"
        else:
            name = f"amem_notes_{embedder.collection_suffix}"[:60]
        self._embedder = embedder
        self._notes = chroma.get_or_create_collection(name, embedding_function=embedder)

        self._note_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._bg_tasks: set = set()
        self._idle_timers: Dict[str, asyncio.Task] = {}

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(self, user_id: str, conv_id: str, role: MsgRole, content: str,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """写入最近对话；用户说的陈述同时存成一条原文笔记。都不调模型。"""
        user_id, conv_id = safe_text(user_id), safe_text(conv_id)
        msg = Message(role=role, content=safe_text(content),
                      metadata={safe_text(k): safe_text(v) for k, v in (metadata or {}).items()})
        if role == MsgRole.USER:
            await self._add_note(user_id, conv_id, msg)
        try:
            await asyncio.wait_for(self._push(user_id, conv_id, msg), self._redis_timeout * 2)
        except Exception as ex:                     # noqa: BLE001 —— 回复已经生成，写记忆失败不能让请求报错
            logger.warning(f"写入最近对话失败，本轮不记忆: {ex}")

    async def _push(self, user_id: str, conv_id: str, msg: Message) -> None:
        key = self._window_key(user_id, conv_id)
        await self._redis.lpush(key, json.dumps({"role": msg.role.value, "content": msg.content,
                                                 "ts": msg.timestamp.isoformat(), "metadata": msg.metadata}))
        await self._redis.ltrim(key, 0, self.WINDOW - 1)
        await self._redis.expire(key, 86400)

    @classmethod
    def worth_noting(cls, text: str) -> bool:
        """太短的（“好的”）和纯提问（带问号、没讲自己的情况）不存：它们不是要记住的事，还会挤掉真正的原话。"""
        if len(text) < cls.NOTE_MIN_CHARS:
            return False
        return not (re.search(r"[?？]\s*$", text) and not has_self_info(text))

    async def _add_note(self, user_id: str, conv_id: str, msg: Message) -> None:
        """先存原文，马上就能被检索到；关键词、链接等后台补全。同一句话再说一遍只刷新时间，不重复存。"""
        text = msg.content.strip()
        if not self.worth_noting(text):
            return
        note_id = hashlib.md5(f"{user_id}\x00{text}".encode()).hexdigest()
        ts = msg.timestamp.isoformat()

        def write() -> None:
            got = self._notes.get(ids=[note_id])
            if got.get("ids"):
                meta = {**(got.get("metadatas") or [{}])[0], "ts": ts, "conv_id": conv_id}
                self._notes.update(ids=[note_id], metadatas=[meta])
                return
            self._notes.add(ids=[note_id], documents=[text], metadatas=[{
                "user_id": user_id, "conv_id": conv_id, "ts": ts, "text": text, "enriched": "0", "tries": 0,
                "keywords": "", "tags": "", "context": "", "event_time": "", "slot": "", "value": "",
                "links": "[]", "versions": "[]", "superseded_by": ""}])

        try:
            await asyncio.wait_for(asyncio.to_thread(write), 10)
        except Exception as ex:                     # noqa: BLE001 —— 少一条笔记不能影响对话
            logger.warning(f"写入笔记失败: {ex}")

    def schedule_after_turn(self, user_id: str, conv_id: str) -> None:
        """对话接口每轮回复后调用：在后台看要不要补全笔记，并重新开始这位用户的空闲计时。不拖慢这一轮。"""
        task = asyncio.create_task(self.after_turn(user_id, conv_id))
        self._bg_tasks.add(task)                    # 保留引用，任务跑完前不会被回收
        task.add_done_callback(self._bg_tasks.discard)
        user_id = safe_text(user_id)
        old = self._idle_timers.pop(user_id, None)
        if old is not None:
            old.cancel()
        timer = asyncio.create_task(self._enrich_when_idle(user_id))
        self._idle_timers[user_id] = timer
        timer.add_done_callback(lambda t, u=user_id: self._idle_timers.pop(u, None) if self._idle_timers.get(u) is t else None)

    async def _enrich_when_idle(self, user_id: str) -> None:
        """用户最后一句话之后 IDLE_SECONDS 秒还没新消息，就把剩下的笔记补全：
        否则他最后几句要等攒够一批才补全，换个会话再问时这几句没有背景、链接和关键事实。"""
        try:
            await asyncio.sleep(self.IDLE_SECONDS)
            await self.enrich_notes(user_id)
        except asyncio.CancelledError:
            pass
        except Exception as ex:                     # noqa: BLE001
            logger.warning(f"空闲补全笔记失败: {ex}")

    async def after_turn(self, user_id: str, conv_id: str) -> None:
        """补全时机：未补全的攒够 NOTE_BATCH 句，或者别的会话留下了没补全的、且那个会话已经停了 IDLE_SECONDS 秒
        （用户换了会话；同时开着几个窗口时不会每轮都触发）。
        滑出最近对话的原文笔记本来就能被检索到，不为它单独调一次模型；用户停下来时由空闲计时补全。"""
        user_id, conv_id = safe_text(user_id), safe_text(conv_id)
        try:
            pending = await self._pending(user_id)
            stale = (datetime.now() - timedelta(seconds=self.IDLE_SECONDS)).isoformat()
            left_behind = any(p["meta"].get("conv_id") != conv_id and p["meta"].get("ts", "") < stale for p in pending)
            if len(pending) < self.NOTE_BATCH and not left_behind:
                return
            await self.enrich_notes(user_id)
        except Exception as ex:                     # noqa: BLE001 —— 后台任务，失败只记日志
            logger.warning(f"补全笔记失败: {ex}")

    async def wait_background(self) -> None:
        """等后台补全跑完（评测和关闭服务时用）。"""
        while self._bg_tasks:
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)

    async def flush_notes(self, user_id: str) -> int:
        """等后台任务结束并补全这位用户的全部笔记（评测和测试用）。"""
        await self.wait_background()
        return await self.enrich_notes(safe_text(user_id))

    async def enrich_notes(self, user_id: str) -> int:
        """补全未补全的笔记，每 NOTE_BATCH 句一次模型调用。返回补全的条数。"""
        done = 0
        async with self._note_locks[user_id]:
            pending = await self._pending(user_id)
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
            hits = await asyncio.to_thread(self._nearest, r["meta"].get("text", ""),
                                           {"$and": [{"user_id": user_id}, {"enriched": "1"}]}, self.NEIGHBORS_EACH)
            for nid, nmeta in hits:
                if nid not in batch_ids and nid not in neighbors and len(neighbors) < self.NEIGHBOR_CAP:
                    neighbors[nid] = nmeta

        m_ids = {f"M{k + 1}": r for k, r in enumerate(batch)}
        n_ids = {f"N{k + 1}": (nid, meta) for k, (nid, meta) in enumerate(neighbors.items())}
        msgs = "\n".join(f"{k}（说于 {r['meta'].get('ts', '')[:10]}）：{r['meta'].get('text', '')}" for k, r in m_ids.items())
        olds = "\n".join(f"{k}（说于 {m.get('ts', '')[:10]}，背景：{m.get('context', '')}）：{m.get('text', '')}"
                         for k, (_, m) in n_ids.items()) or "（无）"
        slots = "；".join(f"{k}（{v}）" for k, v in FACT_SLOTS.items())
        prompt = safe_text(
            "你在整理一位留学咨询用户的记忆笔记。下面 M 开头的是他新说的话，N 开头的是以前的笔记。\n"
            "对每条 M 给出：\n"
            "- keywords：3–6 个以后能用来搜到它的关键词（人名、地点、事件、物品，用原话的语言）；tags：最多 3 个主题标签；\n"
            "- context：一句话背景（这句话在讲什么）；\n"
            "- event_time：话里的事发生在什么时候，按说话日期换算成 YYYY-MM-DD 或 YYYY-MM；话里自带日期的以话里为准；说不出就留空；\n"
            f"- slot 和 value：如果这句话说出了用户**本人**的关键事实，slot 填字段名（{slots}），value 填简短取值"
            "（保留数字和考试名，如“雅思 7.0”“每年 25 万”）；讲别人的事、闲聊、没有明确事实就都留空；\n"
            "- links：和哪些 M/N 讲的是相关的事；supersedes：取代了哪条 N（同一件事后来改了主意或更新了，才算取代）。\n"
            "如果新信息让某条 N 的背景需要补充或更正，在 neighbor_updates 里给出新的背景（只改背景，不改原话）。\n"
            "只返回 JSON："
            '{"notes": [{"m": "M1", "keywords": [], "tags": [], "context": "", "event_time": "", "slot": "", "value": "",'
            ' "links": [], "supersedes": []}], "neighbor_updates": [{"n": "N1", "context": ""}]}\n\n'
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
            slot = str(n.get("slot") or "").strip()
            value = safe_text(n.get("value") or "").strip()[:80]
            if slot not in FACT_SLOTS or not value:
                slot, value = "", ""
            links = [x for x in (resolve(str(ref)) for ref in n.get("links") or []) if x and x != r["id"]]
            for ref in n.get("supersedes") or []:
                old = resolve(str(ref))
                if old and old != r["id"]:
                    superseded[old] = r["id"]
            for target in links:
                link_back[target].append(r["id"])
            meta.update(enriched="1", keywords="、".join(kws), tags="、".join(tags), slot=slot, value=value,
                        context=safe_text(n.get("context") or "")[:200], event_time=str(n.get("event_time") or "")[:20],
                        links=json.dumps(sorted(set(links))))
            await asyncio.to_thread(self._notes.update, ids=[r["id"]], metadatas=[meta], documents=[self._note_doc(meta)])

        # 旧笔记：背景更新（只追加版本）、反向链接、取代标记
        updates = {u.get("n"): safe_text(u.get("context") or "") for u in obj.get("neighbor_updates") or [] if isinstance(u, dict)}
        touched = set(link_back) | set(superseded) | {n_ids[k][0] for k in updates if k in n_ids}
        for nid in touched:
            got = await asyncio.to_thread(self._notes.get, ids=[nid])
            if not got.get("ids"):
                continue
            meta = dict((got.get("metadatas") or [{}])[0] or {})
            meta["links"] = json.dumps(sorted(set(json.loads(meta.get("links") or "[]")) | set(link_back.get(nid, []))))
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

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        """最近对话 + 和这次提问相关的笔记 + 关键事实。"""
        user_id, conv_id, query = safe_text(user_id), safe_text(conv_id), safe_text(query)
        try:
            recent = await asyncio.wait_for(self._get_window(user_id, conv_id), self._redis_timeout * 2)
        except Exception as ex:                     # noqa: BLE001 —— 记忆是背景，读不到就按没有记忆继续
            logger.warning(f"读取最近对话失败，按无记忆继续: {ex}")
            recent = []
        exclude = [m.content for m in recent]
        notes, facts = await asyncio.gather(
            self._search_notes(user_id, query or (recent[-1].content if recent else ""), exclude),
            self._facts(user_id),
        )
        return MemoryContext(recent_messages=recent, notes=notes, facts=facts)

    async def _search_notes(self, user_id: str, query: str, exclude: List[str]) -> List[str]:
        """取相关笔记，顺链接扩展；已经在最近对话里的不重复放。"""
        query = query.strip()
        if not query:
            return []
        skip = {t.strip() for t in exclude}
        try:
            hits = await asyncio.wait_for(asyncio.to_thread(
                self._nearest, query, {"user_id": user_id}, self.NOTE_TOP_K + len(skip)), self.SEARCH_TIMEOUT)
            hits = [(i, m) for i, m in hits if m.get("text", "").strip() not in skip][: self.NOTE_TOP_K]
            have = {i for i, _ in hits}
            want: List[str] = []
            for _, meta in hits:
                for lid in json.loads(meta.get("links") or "[]"):
                    if lid not in have and lid not in want:
                        want.append(lid)
            if want[: self.LINK_EXPAND]:
                got = await asyncio.to_thread(self._notes.get, ids=want[: self.LINK_EXPAND])
                hits += [(i, m or {}) for i, m in zip(got.get("ids") or [], got.get("metadatas") or [])
                         if (m or {}).get("text", "").strip() not in skip]
        except Exception as ex:                     # noqa: BLE001
            logger.warning(f"笔记检索失败: {ex}")
            return []
        hits.sort(key=lambda h: h[1].get("ts", ""))
        out: List[str] = []
        for _, meta in hits:
            line = self._render_note(meta)
            if line not in out:
                out.append(line)
        return out

    def _nearest(self, text: str, where: Dict[str, Any], k: int) -> List[Tuple[str, Dict[str, Any]]]:
        """按向量取最近的 k 条（同步，放在线程里跑）。

        ChromaDB 0.5 的 HNSW 带过滤条件查询时偶尔会报“Cannot return the results in a contigious 2D array”
        （评测里并发写入、查询时出现过，单线程复现不出来），以前这时整次检索返回空。现在改成在这位用户的笔记里精确计算距离：
        单个用户的笔记最多几百条，精确算也很快，结果还更准。
        """
        qvec = self._embed_query(text)
        try:
            res = self._notes.query(query_embeddings=[qvec], n_results=k, where=where, include=["metadatas"])
            return list(zip((res.get("ids") or [[]])[0], [m or {} for m in (res.get("metadatas") or [[]])[0]]))
        except Exception as ex:                     # noqa: BLE001
            if "contig" not in str(ex):
                raise
            got = self._notes.get(where=where, include=["embeddings", "metadatas"])
            embs = got.get("embeddings")            # numpy 数组，不能写 `or []`
            embs = [] if embs is None else list(embs)
            rows = list(zip(got.get("ids") or [], embs, got.get("metadatas") or []))
            rows.sort(key=lambda r: -float(sum(a * b for a, b in zip(qvec, r[1]))))
            return [(i, m or {}) for i, _, m in rows[:k]]

    def _embed_query(self, text: str) -> List[float]:
        if hasattr(self._embedder, "embed_query"):      # bge 系列：查询带检索指令编码
            return list(self._embedder.embed_query(text))
        return [float(x) for x in self._embedder([text])[0]]

    async def _facts(self, user_id: str) -> List[str]:
        """关键事实：同一字段按说话时间排，以最新为准，以前的取值留在括号里。"""
        try:
            got = await asyncio.wait_for(asyncio.to_thread(
                self._notes.get, where={"$and": [{"user_id": user_id}, {"slot": {"$ne": ""}}]}, include=["metadatas"],
            ), self.SEARCH_TIMEOUT)
        except Exception as ex:                     # noqa: BLE001
            logger.warning(f"读取关键事实失败: {ex}")
            return []
        slots: Dict[str, List[str]] = {}
        for meta in sorted((m or {} for m in got.get("metadatas") or []), key=lambda m: m.get("ts", "")):
            values = slots.setdefault(meta.get("slot", ""), [])
            value = (meta.get("value") or "").strip()
            if value and (not values or values[-1] != value):
                values = [v for v in values if v != value] + [value]
                slots[meta["slot"]] = values
        lines = []
        for slot, values in slots.items():
            if slot not in FACT_SLOTS or not values:
                continue
            label = FACT_SLOTS[slot].split("（")[0]
            if slot in LIST_SLOTS:
                lines.append(f"{label}：{'；'.join(values[-LIST_SHOW:])}")
            else:
                earlier = f"（之前：{'、'.join(values[:-1])}）" if len(values) > 1 else ""
                lines.append(f"{label}：{values[-1]}{earlier}")
        return lines

    def _render_note(self, meta: Dict[str, Any]) -> str:
        said = (meta.get("ts") or "")[:10]
        when = f"[{meta['event_time']}｜说于 {said}]" if meta.get("event_time") else (f"[{said}]" if said else "")
        text = (meta.get("text") or "").strip()
        if len(text) > self.NOTE_SHOW_CHARS:
            text = text[: self.NOTE_SHOW_CHARS].rstrip() + "…"
        out = f"{when} {text}".strip()
        if meta.get("context"):
            out += f"（背景：{meta['context']}）"
        if meta.get("superseded_by"):
            out += "（后来有更新，以新的为准）"
        return out

    async def _get_window(self, user_id: str, conv_id: str) -> List[Message]:
        raws = await self._redis.lrange(self._window_key(user_id, conv_id), 0, self.WINDOW - 1)
        msgs = []
        for raw in reversed(raws):                  # lpush 最新在前，反过来还原时序
            d = json.loads(raw)
            msgs.append(Message(role=MsgRole(d["role"]), content=d["content"],
                                timestamp=datetime.fromisoformat(d["ts"]), metadata=d.get("metadata", {})))
        return msgs

    @staticmethod
    def _window_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    async def close(self) -> None:
        """关闭前取消空闲计时、等后台补全最多几秒，再关 Redis 连接。"""
        for timer in list(self._idle_timers.values()):
            timer.cancel()
        try:
            await asyncio.wait_for(self.wait_background(), 5)
        except Exception as ex:                     # noqa: BLE001 —— 没补完的笔记下次有新消息时照样会补
            logger.info(f"关闭时后台补全没跑完: {ex!r}")
        await self._redis.aclose()


def build_memory_manager(**kwargs: Any) -> MemoryManager:
    return MemoryManager(**kwargs)
