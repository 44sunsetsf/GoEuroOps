"""
多轮对话记忆管理。

三层记忆：
  1. 工作记忆（Redis）：当前会话最近的消息原文，满 COMPRESS_AT 条压缩，只留最近 KEEP_RAW 条原文
  2. 情景记忆（ChromaDB）：压缩下来的摘要（附原文存档），按语义检索，可跨会话
  3. 用户画像（ChromaDB）：用户本人说出的事实，按固定字段归类，**只追加、带时间**；
     取用时同一字段以最新为准，旧值留档（“之前：荷兰”），一次提炼错了也不会把对的盖掉

写画像的时机：用户这句话在讲自己的情况（带“我”，且出现数字、专业、成绩、国家、预算这类词）才提炼一次；
压缩时再对被压缩的消息整理一遍，补上漏掉的事实。
提示词里放全部工作记忆（最多 COMPRESS_AT - 1 条），助手的长回复截断，用户的话不截断（事实都在用户的话里）。
Embedding 用 ChromaDB 内置模型，不依赖外部 API。
"""
import hashlib
import asyncio
import os
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import chromadb
import redis.asyncio as redis

from core.llm_utils import NO_THINKING_KWARGS, extract_text_content
from core.llm_utils import make_client, safe_text
from core.config import DEFAULT_MODEL

logger = logging.getLogger(__name__)


ASSISTANT_MSG_CHARS = 300   # 提示词里每条助手回复最多放多少字

# 画像只记用户本人明确说出的事实，按固定字段归类
PROFILE_FIELDS: Dict[str, str] = {
    "major": "本科专业",
    "school": "本科院校",
    "gpa": "GPA / 均分",
    "language": "语言成绩（写明考试名，如 雅思 6.5）",
    "target_country": "目标国家",
    "intake": "计划入学时间",
    "budget": "预算",
    "graduation": "毕业时间 / 工作年限",
    "experience": "实习、工作或项目经历",
    "other": "其他明确的偏好或约束",
}
_SELF_INFO = re.compile(
    r"\d|专业|本科|硕士|毕业|学校|大学|学院|gpa|绩点|均分|雅思|托福|ielts|toefl|gre|预算|万|入学|春季|秋季|冬季|学期|"
    r"实习|工作|经历|项目|决定|改成|改申|不去|打算|想去|计划|目标|"
    r"瑞典|德国|荷兰|芬兰|丹麦|挪威|英国|美国|北欧|欧洲|新加坡|香港|澳洲|加拿大",
    re.IGNORECASE,
)


def has_self_info(text: str) -> bool:
    """这句用户消息是不是在讲自己的情况：带“我”，并且出现数字或专业、成绩、国家、预算这类词。"""
    t = text or ""
    return "我" in t and bool(_SELF_INFO.search(t))


def normalize_profile(raw: Any) -> Dict[str, Any]:
    """统一成 {"version": 2, "facts": [...]}；旧版本的整份画像放进 legacy 原样保留，不丢。"""
    if isinstance(raw, dict) and isinstance(raw.get("facts"), list):
        return {"version": 2, "facts": raw["facts"], **({"legacy": raw["legacy"]} if raw.get("legacy") else {})}
    if isinstance(raw, dict) and raw:
        return {"version": 2, "facts": [], "legacy": raw}
    return {"version": 2, "facts": []}


def profile_view(profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """每个字段的最新值，以及之前出现过的不同取值（按时间先后）。"""
    view: Dict[str, Dict[str, Any]] = {}
    for fact in sorted(profile.get("facts") or [], key=lambda f: str(f.get("ts", ""))):
        name, value = fact.get("field"), str(fact.get("value", "")).strip()
        if not name or not value:
            continue
        slot = view.setdefault(name, {"value": value, "history": []})
        if value != slot["value"]:
            if slot["value"] not in slot["history"]:
                slot["history"].append(slot["value"])
            slot["value"] = value
    return view


def render_profile(profile: Dict[str, Any]) -> str:
    """画像转成给模型看的几行字，中文不转义。"""
    lines = []
    for name, slot in profile_view(profile).items():
        label = PROFILE_FIELDS.get(name, name)
        earlier = f"（之前：{'、'.join(slot['history'])}）" if slot["history"] else ""
        lines.append(f"- {label}：{slot['value']}{earlier}")
    legacy = profile.get("legacy")
    if legacy:
        lines.append(f"- 旧版画像：{json.dumps(legacy, ensure_ascii=False)}")
    return "\n".join(lines)


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
    """传给 Agent 的完整上下文。"""
    recent_messages:  List[Message]   # 工作记忆：最近对话
    relevant_history: List[str]       # 情景记忆：语义相关的历史片段
    user_profile:     Dict[str, Any]  # 用户画像：{"version": 2, "facts": [...]}，见 normalize_profile
    summary:          str             # 当前会话摘要（压缩后）

    @staticmethod
    def _clean(text: str) -> str:
        return safe_text(text)

    def to_prompt_text(self) -> str:
        """将记忆上下文格式化为 LLM 可用的文本。"""
        parts = []
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.relevant_history:
            parts.append("[相关历史]\n" + "\n".join(f"- {self._clean(h)}" for h in self.relevant_history[:3]))
        profile_text = render_profile(normalize_profile(self.user_profile)) if self.user_profile else ""
        if profile_text:
            parts.append(f"[用户画像]（用户本人说过的事实，后说的为准）\n{self._clean(profile_text)}")
        if self.recent_messages:
            parts.append("[最近对话]")
            # 工作记忆里有多少就放多少（压缩前最多 COMPRESS_AT - 1 条），不再只放最后 8 条：
            # 否则 9–14 条时最早几条既不在提示词里、也还没进摘要。助手的长回复截断，用户的话不截断。
            for m in self.recent_messages:
                content = self._clean(m.content)
                if m.role == MsgRole.ASSISTANT and len(content) > ASSISTANT_MSG_CHARS:
                    content = content[:ASSISTANT_MSG_CHARS].rstrip() + "…（略）"
                parts.append(f"{m.role.value}: {content}")
        return "\n\n".join(parts)


class MemoryManager:
    """
    三级记忆管理器。

    工作记忆存 Redis（TTL 24h），情景记忆和用户画像存 ChromaDB（持久化）。
    """

    WORKING_MAX   = 20    # 工作记忆最大条数，超过则触发压缩
    COMPRESS_AT   = 15    # 达到此条数时压缩，保留摘要 + 最近 KEEP_RAW 条
    KEEP_RAW      = 5
    HISTORY_TOP_K = 5     # 情景记忆检索返回条数
    SUMMARY_MAX_CHARS = 800
    PROFILE_DOC_PREFIX = "user_profile:"

    def __init__(
        self,
        redis_url:    str = "redis://localhost:6379/0",
        chroma_host:  str = "localhost",
        chroma_port:  int = 8000,
        chroma_path:  str = "./data/chroma",
        api_key:      str = "",
        base_url:     Optional[str] = None,
        model:        str = DEFAULT_MODEL,
    ):
        self._client = make_client(api_key, base_url, "memory")
        self._model  = model

        # 工作记忆在每次对话的主链路上：Redis 慢或挂了不能拖住对话，所以连接和读写都设超时，
        # 失败时按"没有记忆"继续（见 get_context / add_message）。
        timeout = float(os.getenv("GOEUROOPS_MEMORY_REDIS_TIMEOUT", "1.0"))
        self._redis_timeout = timeout
        self._redis = redis.from_url(
            redis_url, decode_responses=True, socket_timeout=timeout, socket_connect_timeout=timeout,
        )

        # ChromaDB：优先连接独立服务（docker compose 模式），连不上或 chroma_host 为空时用本地嵌入式
        try:
            if not chroma_host:
                raise ConnectionError("未配置 CHROMA_HOST")
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            chroma = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            chroma.heartbeat()  # 测试连接
            logger.info(f"ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
            chroma = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 情景记忆：存储历史对话片段
        self._episodic = chroma.get_or_create_collection("episodic")
        # 用户画像：存储提炼出的偏好和实体
        self._profile  = chroma.get_or_create_collection("user_profile")
        # 画像是“读出来、追加、写回去”，同一用户的两次后台更新不能交错，否则会丢事实
        self._profile_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role:    MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一条消息写入工作记忆，超阈值时自动压缩。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        clean_metadata = {
            self._safe_text(k): self._safe_metadata_value(v)
            for k, v in (metadata or {}).items()
        }
        msg = Message(role=role, content=self._safe_text(content), metadata=clean_metadata)
        key = self._wm_key(user_id, conv_id)

        try:
            length = await asyncio.wait_for(self._append(key, msg), self._redis_timeout * 2)
        except Exception as ex:                     # noqa: BLE001 —— 回复已经生成，写记忆失败不能让请求报错
            logger.warning(f"写入工作记忆失败，本轮不记忆: {ex}")
            return

        # 压缩要调一次模型，不放进上面的超时里：中途取消会停在“旧列表已删、还没写回”的状态
        if length >= self.COMPRESS_AT:
            try:
                await self._compress(user_id, conv_id)
            except Exception as ex:                 # noqa: BLE001 —— 压缩失败下次还会再触发
                logger.warning(f"工作记忆压缩失败: {ex}")

    async def _append(self, key: str, msg: "Message") -> int:
        """追加到 Redis 列表（左推，最新在前），刷新 24h TTL，返回当前条数。"""
        await self._redis.lpush(key, json.dumps({
            "role":      msg.role.value,
            "content":   msg.content,
            "ts":        msg.timestamp.isoformat(),
            "metadata":  msg.metadata,
        }))
        await self._redis.expire(key, 86400)
        return int(await self._redis.llen(key))

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        """对话接口每轮回复后在后台调用。只有用户这句话在讲自己的情况时，才花一次模型调用提炼事实。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        try:
            messages = await asyncio.wait_for(self._get_working_memory(user_id, conv_id), self._redis_timeout * 2)
        except Exception as ex:                     # noqa: BLE001 —— 后台任务，失败只记日志
            logger.warning(f"更新画像时读取工作记忆失败: {ex}")
            return
        last_user = next((m for m in reversed(messages) if m.role == MsgRole.USER), None)
        if last_user is None or not has_self_info(last_user.content):
            return
        await self._extract_facts(user_id, conv_id, messages[-6:])

    async def _extract_facts(self, user_id: str, conv_id: str, messages: List[Message]) -> int:
        """从几条消息里提炼用户本人说出的事实，和已有画像比对后只追加新值。返回追加了几条。"""
        if not messages:
            return 0
        async with self._profile_locks[user_id]:
            profile = await self._get_profile(user_id)
            current = {name: slot["value"] for name, slot in profile_view(profile).items()}
            dialog = "\n".join(
                f"{m.role.value}: {self._safe_text(m.content)[:ASSISTANT_MSG_CHARS] if m.role == MsgRole.ASSISTANT else self._safe_text(m.content)}"
                for m in messages
            )
            fields = "；".join(f"{k}（{v}）" for k, v in PROFILE_FIELDS.items())
            prompt = self._safe_text(f"""你在维护一位留学咨询用户的画像。从下面的对话里，只提取**用户本人明确说出的、关于他自己的事实**。
不要记录用户问了什么、关注什么话题，也不要记录助手说的内容。
字段只能用：{fields}
已有画像（各字段的最新值）：{json.dumps(current, ensure_ascii=False)}
规则：和已有画像相同的不要输出；用户改了说法（换了目标国家、重考了成绩等）就输出新值；值要简短，保留数字和考试名。

对话：
{dialog}

只返回 JSON：{{"facts": [{{"field": "字段名", "value": "值"}}]}}；没有新事实就返回 {{"facts": []}}""")
            try:
                resp = await self._client.messages.create(
                    model=self._model, max_tokens=300, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                    **NO_THINKING_KWARGS,
                )
                data = _parse_json_object(extract_text_content(resp.content))
            except Exception as ex:                 # noqa: BLE001 —— 提炼失败就这次不更新，已有画像原样保留
                logger.warning(f"提炼画像失败，本轮不更新: {ex}")
                return 0

            added: List[Dict[str, Any]] = []
            now = datetime.now().isoformat()
            for item in data.get("facts") or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("field", "")).strip()
                value = self._safe_text(item.get("value", "")).strip()[:80]
                if name not in PROFILE_FIELDS or not value or current.get(name, "").strip() == value:
                    continue
                added.append({"field": name, "value": value, "ts": now, "conv_id": conv_id})
                current[name] = value
            if not added:
                return 0
            profile["facts"].extend(added)
            await self._save_profile(user_id, conv_id, profile)
            logger.info(f"用户画像追加 {len(added)} 条事实: {user_id}")
            return len(added)

    async def _save_profile(self, user_id: str, conv_id: str, profile: Dict[str, Any]) -> None:
        doc_id = self._profile_doc_id(user_id)
        doc_text = self._safe_text(json.dumps(profile, ensure_ascii=False))
        try:
            await asyncio.to_thread(self._profile.delete, ids=[doc_id])
        except Exception:                           # noqa: BLE001 —— 第一次写画像时旧记录本来就不存在
            logger.debug("删除旧画像文档失败（可能本来就不存在）: %s", user_id, exc_info=True)
        await asyncio.to_thread(
            self._profile.add,
            ids=[doc_id],
            documents=[doc_text],
            metadatas=[{"user_id": user_id, "conv_id": conv_id, "updated_at": datetime.now().isoformat()}],
        )

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        """
        构建完整的记忆上下文。

        query 用于从情景记忆中检索语义相关的历史片段。
        """
        # 1. 工作记忆（当前会话最近消息）
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        query = self._safe_text(query)

        try:
            recent = await asyncio.wait_for(self._get_working_memory(user_id, conv_id), self._redis_timeout * 2)
        except Exception as ex:                     # noqa: BLE001 —— 记忆是背景，读不到就按没有记忆继续
            logger.warning(f"读取工作记忆失败，按无记忆继续: {ex}")
            recent = []

        # 2. 情景记忆（跨会话语义检索）
        history = await self._search_episodic(
            user_id,
            conv_id,
            query or (recent[-1].content if recent else ""),
        )

        # 3. 用户画像
        profile = await self._get_profile(user_id)

        # 4. 会话摘要（如果已压缩过）
        try:
            summary = await asyncio.wait_for(
                self._redis.get(self._summary_key(user_id, conv_id)), self._redis_timeout * 2,
            ) or ""
        except Exception as ex:                     # noqa: BLE001
            logger.warning(f"读取会话摘要失败，按无摘要继续: {ex}")
            summary = ""

        return MemoryContext(
            recent_messages=recent,
            relevant_history=history,
            user_profile=profile,
            summary=summary,
        )

    # ── 压缩（防止 context 爆炸）─────────────────────────────────────────────

    async def _compress(self, user_id: str, conv_id: str) -> None:
        """
        工作记忆压缩：
          1. 用 LLM 对旧消息生成摘要
          2. 摘要存 Redis（覆盖旧摘要）
          3. 旧消息存入情景记忆（ChromaDB）供跨会话检索
          4. 工作记忆只保留最近 5 条
        """
        messages = await self._get_working_memory(user_id, conv_id)
        if len(messages) < self.COMPRESS_AT:
            return

        to_compress = messages[:-self.KEEP_RAW]
        keep        = messages[-self.KEEP_RAW:]

        # 压缩前把要压缩的消息再整理一遍画像，补上逐轮提炼漏掉的事实（相当于“空闲时整理”）。
        # 它和写摘要互不依赖，并行执行，压缩这一轮不会因此多等一次模型调用。
        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in to_compress))

        async def consolidate() -> None:
            if any(m.role == MsgRole.USER and has_self_info(m.content) for m in to_compress):
                try:
                    await self._extract_facts(user_id, conv_id, to_compress)
                except Exception as ex:             # noqa: BLE001
                    logger.warning(f"压缩前整理画像失败: {ex}")

        async def summarize() -> str:
            # 用户说过的事实和改变必须保留，一般问答一句带过
            prompt = self._safe_text(
                "把下面这段对话压缩成一段不超过 150 字的摘要。重点保留用户本人说出的事实：专业、院校、成绩、目标国家、"
                "入学时间、预算、经历，以及他做过的决定和改变（写成“先……后改为……”）。助手回答的一般性内容一句带过。"
                f"只输出摘要正文。\n\n{text}"
            )
            try:
                resp = await self._client.messages.create(
                    model=self._model, max_tokens=256, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                    **NO_THINKING_KWARGS,
                )
                return self._safe_text(extract_text_content(resp.content)).strip()
            except Exception:                       # noqa: BLE001
                return f"对话包含 {len(to_compress)} 条消息（摘要生成失败）"

        _, summary = await asyncio.gather(consolidate(), summarize())

        # 存摘要到 Redis
        skey = self._summary_key(user_id, conv_id)
        old_summary = await self._redis.get(skey) or ""
        new_summary = await self._merge_summary(old_summary, summary)
        await self._redis.setex(skey, 86400, new_summary)

        # 旧消息存入情景记忆
        await self._store_episodic(user_id, conv_id, text, summary)

        # 重置工作记忆为最近 5 条
        key = self._wm_key(user_id, conv_id)
        await self._redis.delete(key)
        for m in reversed(keep):
            await self._redis.lpush(key, json.dumps({
                "role": m.role.value, "content": m.content,
                "ts": m.timestamp.isoformat(), "metadata": m.metadata,
            }))
        await self._redis.expire(key, 86400)
        logger.info(f"工作记忆压缩完成: {user_id}/{conv_id}，摘要 {len(summary)} 字")

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    async def _get_working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        key  = self._wm_key(user_id, conv_id)
        raws = await self._redis.lrange(key, 0, self.WORKING_MAX - 1)
        msgs = []
        for raw in reversed(raws):  # Redis lpush 最新在前，reversed 还原时序
            d = json.loads(raw)
            msgs.append(Message(
                role=MsgRole(d["role"]),
                content=d["content"],
                timestamp=datetime.fromisoformat(d["ts"]),
                metadata=d.get("metadata", {}),
            ))
        return msgs

    async def _search_episodic(self, user_id: str, conv_id: str, query: str) -> List[str]:
        """语义检索情景记忆。ChromaDB 内置 embedding，不依赖外部 API。"""
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        try:
            results = await self._query_episodic(
                query_text,
                n_results=self.HISTORY_TOP_K,
                # ChromaDB 规定多个条件要用 $and 包起来，直接写两个键会报错（原来这里每次都失败，返回空）
                where={"$and": [{"user_id": self._safe_text(user_id)}, {"conv_id": self._safe_text(conv_id)}]},
            )
            docs = self._extract_docs(results)
            if len(docs) < self.HISTORY_TOP_K:
                fallback = await self._query_episodic(
                    query_text,
                    n_results=self.HISTORY_TOP_K,
                    where={"user_id": self._safe_text(user_id)},
                )
                docs.extend(self._extract_docs(fallback))
            return self._dedupe_texts(docs)[: self.HISTORY_TOP_K]
        except Exception as ex:
            logger.warning(f"情景记忆检索失败: {ex}")
            return []

    async def _store_episodic(self, user_id: str, conv_id: str, text: str, summary: str) -> None:
        """将压缩后的对话片段存入情景记忆。ChromaDB 内置 embedding，不依赖外部 API。"""
        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            text = self._safe_text(text)
            summary = self._safe_text(summary)
            doc_id = hashlib.md5(f"{user_id}{conv_id}{time.time()}".encode()).hexdigest()
            # 直接传 documents，ChromaDB 内置模型自动生成 embedding
            await asyncio.to_thread(
                self._episodic.add,
                ids=[doc_id],
                documents=[summary],
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat(), "full_text": self._safe_text(text[:4000])}],
            )
        except Exception as ex:
            logger.warning(f"存储情景记忆失败: {ex}")

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """获取用户画像（取最新一条）。"""
        try:
            doc_id = self._profile_doc_id(user_id)
            direct = await asyncio.to_thread(self._profile.get, ids=[doc_id])
            if direct.get("documents"):
                return normalize_profile(json.loads(direct["documents"][0]))

            results = await asyncio.to_thread(self._profile.get, where={"user_id": user_id})
            return normalize_profile(self._latest_profile_from_results(results))
        except Exception as ex:                     # noqa: BLE001 —— 画像只是背景，读不到按没有画像继续
            logger.warning(f"读取用户画像失败，按无画像继续: {ex}")
        return normalize_profile({})

    async def close(self) -> None:
        """关闭异步 Redis 连接。"""
        await self._redis.aclose()

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

    @classmethod
    def _profile_doc_id(cls, user_id: str) -> str:
        return f"{cls.PROFILE_DOC_PREFIX}{user_id}"

    @staticmethod
    def _safe_text(value: Any) -> str:
        """转成 ChromaDB 可接受的普通 UTF-8 字符串。"""
        return safe_text(value)

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """递归清洗 metadata，避免 Redis/ChromaDB 后续读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value

    async def _query_episodic(
        self,
        query_text: str,
        n_results: int,
        where: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._episodic.query,
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )

    @staticmethod
    def _extract_docs(results: Dict[str, Any]) -> List[str]:
        docs = results.get("documents") or []
        if not docs:
            return []
        first = docs[0] if isinstance(docs[0], list) else docs
        return [doc for doc in first if isinstance(doc, str) and doc.strip()]

    @staticmethod
    def _dedupe_texts(values: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for value in values:
            text = value.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    async def _merge_summary(self, old_summary: str, new_summary: str) -> str:
        old_summary = self._safe_text(old_summary).strip()
        new_summary = self._safe_text(new_summary).strip()
        if not old_summary:
            return new_summary[: self.SUMMARY_MAX_CHARS]
        if not new_summary:
            return old_summary[: self.SUMMARY_MAX_CHARS]

        prompt = self._safe_text(
            f"""你是对话摘要器。请把下面两段摘要合并为一段不超过 {self.SUMMARY_MAX_CHARS} 个中文字符的摘要。
保留：用户本人说出的事实（专业、成绩、目标国家、入学时间、预算等）、做过的决定和改变（保留先后顺序）、待办事项、未解决问题。
只输出摘要正文，不要编号，不要解释。

旧摘要:
{old_summary}

新增摘要:
{new_summary}
"""
        )
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                **NO_THINKING_KWARGS,
            )
            merged = self._safe_text(extract_text_content(resp.content)).strip()
            if merged:
                return merged[: self.SUMMARY_MAX_CHARS]
        except Exception as ex:
            logger.warning(f"合并摘要失败，回退为截断拼接: {ex}")

        merged = self._safe_text(f"{old_summary}\n{new_summary}").strip()
        return merged[-self.SUMMARY_MAX_CHARS :]

    @staticmethod
    def _latest_profile_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        if not documents:
            return {}

        candidates: List[tuple[str, Dict[str, Any], str]] = []
        for idx, doc in enumerate(documents):
            if not doc:
                continue
            metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}
            ts = str(metadata.get("updated_at") or metadata.get("ts") or "")
            candidates.append((ts, metadata, doc))

        if not candidates:
            return {}

        candidates.sort(key=lambda item: item[0], reverse=True)
        latest_doc = candidates[0][2]
        try:
            return json.loads(latest_doc)
        except Exception:
            return {}
