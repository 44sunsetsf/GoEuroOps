"""
GoEuroOps 的记忆模块，按客服系统常见的分层设计：

  1. 会话状态（Redis，24 小时过期）：当前会话的消息，提示词里放最近 WINDOW 条，最多存 TRANSCRIPT_MAX 条给摘要用。
  2. 客户档案（Redis，profile.py）：用户本人的关键事实，带出处和以前的取值；顾问改过的字段模型不能覆盖。
  3. 咨询记录（Redis，cases.py）：每个会话停下来后写一份摘要（想解决什么、没解决什么、约好的下一步），
     回访时放进提示词，转顾问时附在交接单上。
  4. 原话笔记（ChromaDB，notes.py）：用户说过的每句陈述原样存一条，马上可检索；整理时按 A-Mem 补上关键词、背景、
     链接和取代关系，检索时顺着链接多取几条。保留 RETENTION_DAYS 天。

业务事实以系统记录为准：报价、线索、付款、退款只从工具和线索库取，记忆里只有“用户自己说的”，提示词里也这样标明。
身份证号、护照号在写入前就换成占位符，哪一层都不存。

写入不调模型；整理（补全笔记 + 抽档案候选 + 写咨询记录）在后台一次模型调用完成，时机是：攒够 BATCH 句、用户换了会话、
或者用户停下来 IDLE_SECONDS 秒。任何一步失败都按“少一点记忆”继续，不影响对话。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import chromadb
import redis.asyncio as redis

from business.lead_store import redact_sensitive_ids
from core import chat_log
from core.config import DEFAULT_MODEL
from core.llm_utils import NO_THINKING_KWARGS, extract_text_content, make_client, safe_text
from memory import cases as cases_mod
from memory import notes as notes_mod
from memory import profile as profile_mod
from memory.cases import CaseStore
from memory.consolidate import build_prompt, parse_json_object, transcript_text
from memory.notes import NoteStore, resolve
from memory.profile import ProfileStore, apply_candidate

logger = logging.getLogger(__name__)

ASSISTANT_MSG_CHARS = 300   # 提示词里每条助手回复最多放多少字

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
    facts:           List[str] = field(default_factory=list)  # 客户档案，一行一个字段
    cases:           List[str] = field(default_factory=list)  # 以前的咨询记录（最近的在前）
    max_chars:       int = 6000

    def to_prompt_text(self) -> str:
        """超出字数预算时先去掉最早的原话，再去掉较早的咨询记录；档案和最近对话不裁。"""
        notes, cases = list(self.notes), list(self.cases)
        while True:
            text = self._render(notes, cases)
            if len(text) <= self.max_chars or not (notes or cases):
                return text
            if notes:
                notes.pop(0)
            else:
                cases.pop()

    def _render(self, notes: List[str], cases: List[str]) -> str:
        parts = []
        if self.facts:
            parts.append("[客户档案]（用户自己说的，未核实；付款、订单、报价以系统查询结果为准；括号里是以前的说法）\n"
                         + "\n".join(f"- {safe_text(f)}" for f in self.facts))
        if cases:
            parts.append("[以前的咨询记录]（最近的在前）\n" + "\n".join(f"- {safe_text(c)}" for c in cases))
        if notes:
            parts.append("[用户以前说过的相关原话]（按说话时间先后）\n" + "\n".join(f"- {safe_text(n)}" for n in notes))
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
    TRANSCRIPT_MAX  = 60     # 每个会话最多存多少条消息（写咨询记录用）
    BATCH           = 8      # 攒够多少句未整理的原话就整理一次
    TOP_K           = 6      # 检索取几条笔记
    LINK_EXPAND     = 3      # 顺链接最多多取几条
    NEIGHBORS_EACH  = 3      # 整理时每句取几条已有笔记做候选（链接、取代、理解指代）
    NEIGHBOR_CAP    = 10     # 一次最多给模型看几条旧笔记
    MAX_TRIES       = 2      # 整理失败几次后放弃（原话仍然可检索）
    NOTE_MIN_CHARS  = 6      # 太短的话（“好的”“谢谢”）不存
    CASES_SHOWN     = 2      # 提示词里放最近几份咨询记录
    SEARCH_TIMEOUT  = 5.0    # 提问时检索笔记最多等多久（秒）
    IDLE_SECONDS    = 120    # 用户停下来这么久没说话，就整理并给停下来的会话写咨询记录

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

        # 会话状态在每次对话的主链路上：Redis 慢或挂了不能拖住对话，连接和读写都设超时，失败按“没有记忆”继续
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

        # 向量模型和知识库共用一个（默认 bge-small-zh）；ChromaDB 默认的英文模型对中文几乎没有区分度
        if embedder is _UNSET:
            from retrieval.embeddings import get_shared_embedding_function
            embedder = get_shared_embedding_function()
        if embedder is None:                        # 中文模型加载失败：退回 ChromaDB 默认模型，记忆照常可用
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            embedder, name = DefaultEmbeddingFunction(), "amem_notes"
        else:                                       # 和 A-Mem 版用同一个集合，线上已有的笔记照样能检索
            name = f"amem_notes_{embedder.collection_suffix}"[:60]
        self._notes = NoteStore(chroma.get_or_create_collection(name, embedding_function=embedder), embedder)

        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._bg_tasks: set = set()
        self._idle_timers: Dict[str, asyncio.Task] = {}

    # 档案和咨询记录跟着 self._redis 走（评测和测试会把它换成进程内的替身）
    @property
    def _profiles(self) -> ProfileStore:
        return ProfileStore(self._redis)

    @property
    def _cases(self) -> CaseStore:
        return CaseStore(self._redis)

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(self, user_id: str, conv_id: str, role: MsgRole, content: str,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """写入会话状态；用户说的陈述同时存成一条原话笔记。都不调模型。证件号先换成占位符。"""
        user_id, conv_id = safe_text(user_id), safe_text(conv_id)
        msg = Message(role=role, content=redact_sensitive_ids(safe_text(content)),
                      metadata={safe_text(k): safe_text(v) for k, v in (metadata or {}).items()})
        if role == MsgRole.USER and self.worth_noting(msg.content.strip()):
            try:
                await asyncio.wait_for(asyncio.to_thread(
                    self._notes.add, user_id, conv_id, msg.content.strip(), msg.timestamp), 10)
            except Exception as ex:                 # noqa: BLE001 —— 少一条笔记不能影响对话
                logger.warning(f"写入笔记失败: {ex}")
        try:
            await asyncio.wait_for(self._push(user_id, conv_id, msg), self._redis_timeout * 2)
        except Exception as ex:                     # noqa: BLE001 —— 回复已经生成，写记忆失败不能让请求报错
            logger.warning(f"写入会话失败，本轮不记忆: {ex}")

    async def _push(self, user_id: str, conv_id: str, msg: Message) -> None:
        key = self._window_key(user_id, conv_id)
        await self._redis.lpush(key, json.dumps({"role": msg.role.value, "content": msg.content,
                                                 "ts": msg.timestamp.isoformat(), "metadata": msg.metadata}))
        await self._redis.ltrim(key, 0, self.TRANSCRIPT_MAX - 1)
        await self._redis.expire(key, 86400)
        await self._redis.sadd(self._open_key(user_id), conv_id)     # 还没写进咨询记录的会话
        await self._redis.expire(self._open_key(user_id), 7 * 86400)

    @classmethod
    def worth_noting(cls, text: str) -> bool:
        """太短的（“好的”）和纯提问（带问号、没讲自己的情况）不存：它们不是要记住的事，还会挤掉真正的原话。"""
        if len(text) < cls.NOTE_MIN_CHARS:
            return False
        return not (re.search(r"[?？]\s*$", text) and not has_self_info(text))

    def schedule_after_turn(self, user_id: str, conv_id: str) -> None:
        """对话接口每轮回复后调用：在后台看要不要整理，并重新开始这位用户的空闲计时。不拖慢这一轮。"""
        self._spawn(self.after_turn(user_id, conv_id))
        user_id = safe_text(user_id)
        old = self._idle_timers.pop(user_id, None)
        if old is not None:
            old.cancel()
        timer = asyncio.create_task(self._consolidate_when_idle(user_id))
        self._idle_timers[user_id] = timer
        timer.add_done_callback(lambda t, u=user_id: self._idle_timers.pop(u, None) if self._idle_timers.get(u) is t else None)

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)                    # 保留引用，任务跑完前不会被回收
        task.add_done_callback(self._bg_tasks.discard)

    async def _consolidate_when_idle(self, user_id: str) -> None:
        """用户最后一句话之后 IDLE_SECONDS 秒还没新消息：相当于会话结束，整理剩下的原话并写咨询记录。"""
        try:
            await asyncio.sleep(self.IDLE_SECONDS)
            await self.consolidate(user_id)
        except asyncio.CancelledError:
            pass
        except Exception as ex:                     # noqa: BLE001
            logger.warning(f"空闲整理失败: {ex}")

    async def after_turn(self, user_id: str, conv_id: str) -> None:
        """整理时机：未整理的原话攒够 BATCH 句，或者别的会话已经停了 IDLE_SECONDS 秒（用户换了会话）。
        当前会话不写咨询记录，等它停下来再写。"""
        user_id, conv_id = safe_text(user_id), safe_text(conv_id)
        try:
            pending = await asyncio.to_thread(self._notes.pending, user_id)
            stale = (datetime.now() - timedelta(seconds=self.IDLE_SECONDS)).isoformat()
            left = []
            for other in await self._redis.smembers(self._open_key(user_id)):
                if other == conv_id:
                    continue
                last = await self._redis.lrange(self._window_key(user_id, other), 0, 0)
                if not last or json.loads(last[0]).get("ts", "") < stale:
                    left.append(other)
            if len(pending) >= self.BATCH or left:
                await self.consolidate(user_id, close=left)
        except Exception as ex:                     # noqa: BLE001 —— 后台任务，失败只记日志
            logger.warning(f"整理记忆失败: {ex}")

    async def wait_background(self) -> None:
        """等后台整理跑完（评测和关闭服务时用）。"""
        while self._bg_tasks:
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)

    async def consolidate(self, user_id: str, close: Optional[List[str]] = None) -> int:
        """整理未整理的原话（每 BATCH 句一次模型调用），并给要结束的会话写咨询记录（放在最后一次调用里）。
        close 为 None 时结束这位用户所有还开着的会话（空闲时）。返回整理的原话条数。"""
        user_id = safe_text(user_id)
        done = 0
        async with self._locks[user_id]:
            pending = await asyncio.to_thread(self._notes.pending, user_id)
            convs = list(await self._redis.smembers(self._open_key(user_id))) if close is None else close
            case_inputs = [c for c in [await self._case_input(user_id, conv) for conv in sorted(convs)] if c]
            batches = [pending[i:i + self.BATCH] for i in range(0, len(pending), self.BATCH)] or ([[]] if case_inputs else [])
            for i, batch in enumerate(batches):
                done += await self._consolidate_batch(user_id, batch, case_inputs if i == len(batches) - 1 else [])
            cutoff = datetime.now() - timedelta(days=notes_mod.RETENTION_DAYS)
            try:
                await asyncio.to_thread(self._notes.purge_older_than, user_id, cutoff)
            except Exception as ex:                 # noqa: BLE001
                logger.warning(f"清理过期笔记失败: {ex}")
        return done

    async def _case_input(self, user_id: str, conv_id: str) -> Optional[Dict[str, Any]]:
        """这个会话上次写摘要之后的新消息；没有新消息就不用写，顺手从“还开着”里去掉。"""
        raws = await self._redis.lrange(self._window_key(user_id, conv_id), 0, self.TRANSCRIPT_MAX - 1)
        msgs = [json.loads(r) for r in reversed(raws)]
        prev = await self._cases.get(user_id, conv_id)
        upto = (prev or {}).get("upto", "")
        new = [m for m in msgs if m.get("ts", "") > upto]
        if not new:
            await self._redis.srem(self._open_key(user_id), conv_id)
            return None
        previous = {k: prev[k] for k in ("summary", "open_items", "next_step")} if prev else None
        return {"conv_id": conv_id, "previous": previous, "transcript": transcript_text(new), "upto": new[-1]["ts"]}

    async def _consolidate_batch(self, user_id: str, batch: List[Any], case_inputs: List[Dict[str, Any]]) -> int:
        batch_ids = {nid for nid, _ in batch}
        neighbors: Dict[str, Dict[str, Any]] = {}
        for _, meta in batch:
            if len(neighbors) >= self.NEIGHBOR_CAP:
                break
            hits = await asyncio.to_thread(self._notes.nearest, meta.get("text", ""),
                                           {"$and": [{"user_id": user_id}, {"consolidated": "1"}]}, self.NEIGHBORS_EACH)
            for nid, nmeta in hits:
                if nid not in batch_ids and nid not in neighbors and len(neighbors) < self.NEIGHBOR_CAP:
                    neighbors[nid] = nmeta

        profile = await self._profiles.get(user_id)
        prompt, m_ids, n_ids, c_ids = build_prompt(profile_mod.render(profile), list(neighbors.items()), batch, case_inputs)
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=3000, temperature=0.0,
                messages=[{"role": "user", "content": prompt}], **NO_THINKING_KWARGS,
            )
            obj = parse_json_object(extract_text_content(resp.content))
        except Exception as ex:                     # noqa: BLE001 —— 原话保留，下次再整理；多次失败就放弃
            logger.warning(f"整理记忆的模型输出无效: {ex}")
            for nid, meta in batch:
                tries = int(meta.get("tries") or 0) + 1
                await asyncio.to_thread(self._notes.update, nid,
                                        {**meta, "tries": tries, "consolidated": "1" if tries >= self.MAX_TRIES else "0"})
            return 0

        # 新笔记：补全字段；档案候选交给 apply_candidate
        notes = {n.get("m"): n for n in obj.get("notes") or [] if isinstance(n, dict)}
        link_back: Dict[str, List[str]] = defaultdict(list)
        superseded: Dict[str, str] = {}
        changed = False
        for key, (nid, meta) in zip(m_ids, batch):
            n = notes.get(key) or {}
            for fact in n.get("facts") or []:
                if isinstance(fact, dict):
                    changed |= apply_candidate(profile, str(fact.get("slot") or "").strip(), str(fact.get("value") or ""),
                                               meta.get("text", ""), meta.get("ts", ""))
            links = [x for x in (resolve(str(r), m_ids, n_ids) for r in n.get("links") or []) if x and x != nid]
            for ref in n.get("supersedes") or []:
                old = resolve(str(ref), m_ids, n_ids)
                if old and old != nid:
                    superseded[old] = nid
            for target in links:
                link_back[target].append(nid)
            meta = {**meta, "consolidated": "1",
                    "keywords": "、".join(str(k) for k in (n.get("keywords") or [])[:6]),
                    "tags": "、".join(str(t) for t in (n.get("tags") or [])[:3]),
                    "context": safe_text(n.get("context") or "")[:200],
                    "event_time": str(n.get("event_time") or "")[:20],
                    "links": json.dumps(sorted(set(links) | set(json.loads(meta.get("links") or "[]"))))}
            await asyncio.to_thread(self._notes.update, nid, meta, True)
        if changed:
            await self._profiles.save(user_id, profile)

        # 旧笔记：反向链接、取代标记、背景更正（只追加版本）
        updates = {u.get("n"): safe_text(u.get("context") or "").strip()
                   for u in obj.get("neighbor_updates") or [] if isinstance(u, dict)}
        new_ctx = {n_ids[k]: v for k, v in updates.items() if k in n_ids and v}
        touched = (set(link_back) | set(superseded) | set(new_ctx)) - batch_ids
        for nid, meta in await asyncio.to_thread(self._notes.get, sorted(touched)) if touched else []:
            meta = dict(meta)
            meta["links"] = json.dumps(sorted(set(json.loads(meta.get("links") or "[]")) | set(link_back.get(nid, []))))
            if nid in superseded:
                meta["superseded_by"] = superseded[nid]
            reembed = False
            if new_ctx.get(nid) and new_ctx[nid] != meta.get("context", ""):
                versions = json.loads(meta.get("versions") or "[]")
                versions.append({"context": meta.get("context", ""), "replaced_at": batch[-1][1].get("ts", "")})
                meta.update(versions=json.dumps(versions, ensure_ascii=False), context=new_ctx[nid][:200])
                reembed = True
            await asyncio.to_thread(self._notes.update, nid, meta, reembed)

        # 咨询记录：模型没给的会话留着，下次再写
        by_conv = {c["conv_id"]: c for c in case_inputs}
        for raw in obj.get("cases") or []:
            if not isinstance(raw, dict) or raw.get("c") not in c_ids:
                continue
            item = cases_mod.clean(raw)
            if item is None:
                continue
            conv_id = c_ids[raw["c"]]
            await self._cases.upsert(user_id, {"conv_id": conv_id, "updated_at": by_conv[conv_id]["upto"],
                                               "upto": by_conv[conv_id]["upto"], **item})
            await self._redis.srem(self._open_key(user_id), conv_id)
        return len(batch)

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        """最近对话 + 客户档案 + 以前的咨询记录 + 和这次提问相关的原话。"""
        user_id, conv_id, query = safe_text(user_id), safe_text(conv_id), safe_text(query)
        try:
            recent = await asyncio.wait_for(self._get_window(user_id, conv_id), self._redis_timeout * 2)
        except Exception as ex:                     # noqa: BLE001 —— 记忆是背景，读不到就按没有记忆继续
            logger.warning(f"读取最近对话失败，按无记忆继续: {ex}")
            recent = []
        exclude = [m.content for m in recent]
        notes, facts, cases = await asyncio.gather(
            self._search_notes(user_id, query or (recent[-1].content if recent else ""), exclude),
            self._guard(self._facts(user_id), "读取客户档案"),        # 档案和咨询记录在 Redis：和最近对话一样的超时
            self._guard(self._recent_cases(user_id, conv_id), "读取咨询记录"),
        )
        return MemoryContext(recent_messages=recent, notes=notes, facts=facts, cases=cases)

    async def _guard(self, coro: Any, what: str) -> List[str]:
        try:
            return await asyncio.wait_for(coro, self._redis_timeout * 2)
        except Exception as ex:                     # noqa: BLE001
            logger.warning(f"{what}失败: {ex}")
            return []

    async def _facts(self, user_id: str) -> List[str]:
        return profile_mod.render(await self._profiles.get(user_id))

    async def _recent_cases(self, user_id: str, conv_id: str) -> List[str]:
        return [cases_mod.render(c) for c in await self._cases.recent(user_id, self.CASES_SHOWN, exclude_conv=conv_id)]

    async def _search_notes(self, user_id: str, query: str, exclude: List[str]) -> List[str]:
        """取相关笔记，顺链接扩展；已经在最近对话里的不重复放。"""
        query = query.strip()
        if not query:
            return []
        skip = {t.strip() for t in exclude}
        try:
            hits = await asyncio.wait_for(asyncio.to_thread(
                self._notes.nearest, query, {"user_id": user_id}, self.TOP_K + len(skip)), self.SEARCH_TIMEOUT)
            hits = [(i, m) for i, m in hits if m.get("text", "").strip() not in skip][: self.TOP_K]
            have = {i for i, _ in hits}
            want: List[str] = []
            for _, meta in hits:
                for lid in json.loads(meta.get("links") or "[]"):
                    if lid not in have and lid not in want:
                        want.append(lid)
            if want[: self.LINK_EXPAND]:
                got = await asyncio.to_thread(self._notes.get, want[: self.LINK_EXPAND])
                hits += [(i, m) for i, m in got if m.get("text", "").strip() not in skip]
        except Exception as ex:                     # noqa: BLE001
            logger.warning(f"笔记检索失败: {ex}")
            return []
        hits.sort(key=lambda h: h[1].get("ts", ""))
        out: List[str] = []
        for _, meta in hits:
            line = notes_mod.render(meta)
            if line not in out:
                out.append(line)
        return out

    async def _get_window(self, user_id: str, conv_id: str) -> List[Message]:
        raws = await self._redis.lrange(self._window_key(user_id, conv_id), 0, self.WINDOW - 1)
        msgs = []
        for raw in reversed(raws):                  # lpush 最新在前，反过来还原时序
            d = json.loads(raw)
            msgs.append(Message(role=MsgRole(d["role"]), content=d["content"],
                                timestamp=datetime.fromisoformat(d["ts"]), metadata=d.get("metadata", {})))
        return msgs

    # ── 顾问和隐私 ────────────────────────────────────────────────────────────

    async def get_profile(self, user_id: str) -> Dict[str, Any]:
        """档案原样返回（带出处和历史），给顾问后台看。"""
        return await self._profiles.get(safe_text(user_id))

    async def set_profile_field(self, user_id: str, slot: str, value: str) -> Dict[str, Any]:
        """顾问核实后改一个字段：以后模型不能覆盖它，用户的不同说法记成待核实。"""
        if slot not in profile_mod.FACT_SLOTS:
            raise ValueError(f"未知字段 {slot}，可选：{', '.join(profile_mod.FACT_SLOTS)}")
        user_id = safe_text(user_id)
        async with self._locks[user_id]:
            profile = await self._profiles.get(user_id)
            apply_candidate(profile, slot, value, "顾问填写", datetime.now().isoformat(), source="advisor")
            await self._profiles.save(user_id, profile)
        return profile

    async def forget_user(self, user_id: str) -> Dict[str, int]:
        """隐私删除：清掉这位用户的会话、档案、咨询记录、笔记和后台对话记录。"""
        user_id = safe_text(user_id)
        timer = self._idle_timers.pop(user_id, None)
        if timer is not None:
            timer.cancel()
        async with self._locks[user_id]:
            pattern = re.sub(r"([*?\[\]\\])", r"\\\1", f"wm:{user_id}:") + "*"
            windows = [k async for k in self._redis.scan_iter(match=pattern)]
            for key in windows:
                await self._redis.delete(key)
            await self._redis.delete(self._open_key(user_id))
            await self._profiles.delete(user_id)
            await self._cases.delete(user_id)
            await asyncio.to_thread(self._notes.delete_user, user_id)
            logs = await chat_log.forget(user_id, self._redis)
        logger.info(f"已删除用户记忆: conversations={len(windows)} chatlog={logs}")
        return {"conversations": len(windows), "chatlog_entries": logs}

    # ── 其他 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _window_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _open_key(user_id: str) -> str:
        return f"mem:open:{user_id}"

    async def close(self) -> None:
        """关闭前取消空闲计时、等后台整理最多几秒，再关 Redis 连接。"""
        for timer in list(self._idle_timers.values()):
            timer.cancel()
        try:
            await asyncio.wait_for(self.wait_background(), 5)
        except Exception as ex:                     # noqa: BLE001 —— 没整理完的原话下次有新消息时照样会整理
            logger.info(f"关闭时后台整理没跑完: {ex!r}")
        await self._redis.aclose()


def build_memory_manager(**kwargs: Any) -> MemoryManager:
    return MemoryManager(**kwargs)
