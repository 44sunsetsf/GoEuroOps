"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic
from core.llm_usage import track

from core.llm_utils import NO_THINKING_KWARGS, extract_text_content
from core.text_embedding import cosine, hashed_ngram_embedding

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    """指北工作室的意图体系：通用大类 + 细粒度业务意图（细粒度优先）。"""
    QUERY      = "query"       # 一般信息查询
    COMPLAINT  = "complaint"   # 投诉不满
    REQUEST    = "request"     # 请求操作
    GREETING   = "greeting"    # 问候
    ESCALATION = "escalation"  # 要求升级/找创始人
    STUDY_CONSULT = "study_consult"  # 五国 CS 硕士公开知识咨询
    BILLING    = "billing"     # 费用/付款大类
    ACCOUNT    = "account"     # 个人资料与联系方式
    FEEDBACK   = "feedback"    # 正面反馈
    SERVICE_PROGRESS = "service_progress"  # 已购服务进度（文书改到第几轮、报告何时交付）
    BOOKING = "booking"                    # 预约/改期/取消咨询
    REFUND = "refund"                      # 服务退款
    INVOICE = "invoice"                    # 发票
    PAYMENT_ISSUE = "payment_issue"        # 定金/尾款/付款异常
    DATA_PRIVACY = "data_privacy"          # 资料删除、隐私与授权
    APPLICATION_PROCESS = "application_process"  # 申请材料/截止日期/语言成绩等流程问题
    SERVICE_INQUIRY = "service_inquiry"    # 工作室服务/价格/报价
    HUMAN_HANDOFF = "human_handoff"        # 转人工顾问
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    intent_group: str
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float
    source_scores: Dict[str, float] = field(default_factory=dict)


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:      ["你们工作室在哪里？", "你们是做什么的？", "顾问都是什么背景？"],
    IntentCategory.COMPLAINT:  ["文书返回太慢了！", "说好的时间又拖了", "消息一直没人回"],
    IntentCategory.REQUEST:    ["帮我把联系方式改一下", "我需要一份服务协议", "请把报告再发我一次"],
    IntentCategory.GREETING:   ["你好", "嗨，有人吗", "早上好"],
    IntentCategory.ESCALATION: ["我要投诉你们的服务", "找你们负责人", "我要和创始人直接谈"],
    IntentCategory.STUDY_CONSULT: ["瑞典有哪些英语授课的CS硕士？", "德国和荷兰的CS硕士有什么区别？", "芬兰读计算机硕士好不好申请？"],
    IntentCategory.BILLING:    ["费用怎么付？", "付款相关问题", "我想了解收费方式"],
    IntentCategory.ACCOUNT:    ["修改我的联系方式", "更新我的邮箱", "换个微信号联系"],
    IntentCategory.FEEDBACK:   ["顾问特别专业！", "非常满意", "文书改得很好，谢谢"],
    IntentCategory.SERVICE_PROGRESS: ["我的PS改到第几轮了？", "选校报告什么时候能交付？", "我的文书进度怎么样了"],
    IntentCategory.BOOKING: ["我想预约一次咨询", "能改一下咨询时间吗？", "我要取消明天的咨询"],
    IntentCategory.REFUND: ["我想申请退款", "定金能退吗？", "服务还没开始可以退款吗？"],
    IntentCategory.INVOICE: ["能开发票吗？", "发票抬头怎么改？", "发票什么时候开？"],
    IntentCategory.PAYMENT_ISSUE: ["尾款付不了", "付款失败了", "我好像多付了一笔钱"],
    IntentCategory.DATA_PRIVACY: ["请删除我的个人资料", "你们会把我的信息给别人吗？", "我不想再被联系了"],
    IntentCategory.APPLICATION_PROCESS: ["申请材料需要准备什么？", "雅思一般要多少分？", "APS要提前多久办？"],
    IntentCategory.SERVICE_INQUIRY: ["你们提供什么服务？", "选校咨询怎么收费？", "全程陪跑多少钱？"],
    IntentCategory.HUMAN_HANDOFF: ["转人工", "我要找真人顾问", "请让顾问联系我"],
}

_SPECIFIC_INTENTS = {
    IntentCategory.SERVICE_PROGRESS,
    IntentCategory.BOOKING,
    IntentCategory.REFUND,
    IntentCategory.INVOICE,
    IntentCategory.PAYMENT_ISSUE,
    IntentCategory.DATA_PRIVACY,
    IntentCategory.APPLICATION_PROCESS,
    IntentCategory.SERVICE_INQUIRY,
    IntentCategory.HUMAN_HANDOFF,
}

_GENERIC_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.BILLING,
    IntentCategory.STUDY_CONSULT,
    IntentCategory.ACCOUNT,
    IntentCategory.ESCALATION,
}

_INTENT_GROUPS: Dict[IntentCategory, IntentCategory] = {
    IntentCategory.SERVICE_PROGRESS: IntentCategory.QUERY,
    IntentCategory.BOOKING: IntentCategory.STUDY_CONSULT,
    IntentCategory.REFUND: IntentCategory.BILLING,
    IntentCategory.INVOICE: IntentCategory.BILLING,
    IntentCategory.PAYMENT_ISSUE: IntentCategory.BILLING,
    IntentCategory.DATA_PRIVACY: IntentCategory.ACCOUNT,
    IntentCategory.APPLICATION_PROCESS: IntentCategory.STUDY_CONSULT,
    IntentCategory.SERVICE_INQUIRY: IntentCategory.STUDY_CONSULT,
    IntentCategory.HUMAN_HANDOFF: IntentCategory.ESCALATION,
}



_UNSET = object()


def _group_of(intent: IntentCategory) -> IntentCategory:
    return _INTENT_GROUPS.get(intent, intent)


# 分歧检测阈值：向量和关键词两路都达到时，才允许否决 LLM。
# 否决的前提是这一路比 LLM（基准上约 92%）更准，所以按各自的分数分布校准：
# bge 相似度整体偏高，≥0.8 时模板匹配精确率 95%；字符 n-gram 分数低，用 0.6。
_CONFLICT_MIN_EMB_BGE = 0.8
_CONFLICT_MIN_EMB_NGRAM = 0.6
_CONFLICT_MIN_PAT = 0.5
_PRAGMATIC_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.REQUEST,
    IntentCategory.GREETING,
    IntentCategory.FEEDBACK,
    IntentCategory.COMPLAINT,
    IntentCategory.OTHER,
}

# 紧急关键词
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻"],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "hurry", "now"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点"],
}


_cosine = cosine


def _keyword_in(keyword: str, text: str) -> bool:
    """中文关键词做子串匹配；纯字母关键词按单词边界匹配，避免 "cs" 命中 "docs"。"""
    if keyword.isascii() and keyword[:1].isalnum():
        return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) is not None
    return keyword in text


class IntentRecognizer:
    """
    端到端意图识别器。

    初始化时不加载任何本地模型，所有 AI 能力通过 Anthropic API 调用。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = track(AsyncAnthropic(**kwargs), "intent")
        self.model     = model
        self.threshold = confidence_threshold
        # 本地字符 n-gram 向量始终可用；如果未来客户端暴露 embeddings 资源，
        # _embed_text 会优先尝试远端向量，否则自动回退本地向量。
        self._embedding_enabled = True
        self._semantic: Any = _UNSET

        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message, history)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 和 Embedding 并行（Embedding 不可用时跳过）
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent, confidence, source_scores = self._vote(llm, emb, pat)
        entities = self._extract_entities(message)
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            intent_group=self._intent_group(intent),
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)  # 下次重新计算
            self._cache.clear()  # 模板更新后旧缓存可能对应过时结果
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是留学咨询工作室「指北」的意图分析专家。工作室提供瑞典/德国/荷兰/芬兰/丹麦英语授课 CS 硕士的选校咨询和文书辅导服务。
根据示例判断用户意图，返回 JSON。能匹配细粒度业务意图时优先返回细粒度意图，而不是宽泛大类：
服务内容/价格/报价 → service_inquiry；预约/改期/取消咨询 → booking；已购服务的进度 → service_progress；
申请材料/截止日期/语言成绩/APS → application_process；退款/定金能否退 → refund；发票 → invoice；
付款失败/多付 → payment_issue；删除资料/隐私 → data_privacy；要求真人顾问 → human_handoff。

示例:
{examples}

        {ctx}
        用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
                **NO_THINKING_KWARGS,
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        specific_patterns = {
            IntentCategory.HUMAN_HANDOFF: ["转人工", "真人", "人工顾问", "让顾问联系"],
            IntentCategory.DATA_PRIVACY: ["删除我的", "删除资料", "个人信息", "隐私", "不要再联系", "gdpr"],
            IntentCategory.SERVICE_PROGRESS: ["第几轮", "进度", "什么时候交付", "改好了吗", "报告什么时候"],
            IntentCategory.BOOKING: ["预约", "改期", "改时间", "取消咨询", "约个时间", "booking"],
            IntentCategory.REFUND: ["退款", "退钱", "能退吗", "refund"],
            IntentCategory.INVOICE: ["发票", "抬头", "税号", "invoice"],
            IntentCategory.PAYMENT_ISSUE: ["付款失败", "付不了", "多付", "重复付款", "扣了两次", "payment failed"],
            IntentCategory.APPLICATION_PROCESS: ["申请材料", "截止", "雅思", "托福", "语言成绩", "aps", "uni-assist", "推荐信要", "deadline", "requirement"],
            IntentCategory.SERVICE_INQUIRY: ["收费", "价格", "多少钱", "报价", "套餐", "陪跑", "服务内容", "优惠", "便宜", "price"],
        }
        generic_patterns = {
            IntentCategory.ESCALATION: ["投诉", "负责人", "创始人"],
            IntentCategory.COMPLAINT:  ["太慢", "太差", "拖了", "没人回", "不满意"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "哪里"],
            IntentCategory.REQUEST:    ["帮我", "需要", "please", "help"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
            IntentCategory.FEEDBACK:   ["谢谢", "满意", "专业", "很棒"],
            IntentCategory.BILLING:    ["付款", "定金", "尾款", "费用"],
            IntentCategory.STUDY_CONSULT: ["瑞典", "德国", "荷兰", "芬兰", "丹麦", "北欧", "硕士", "研究生", "cs", "计算机"],
            IntentCategory.ACCOUNT:    ["联系方式", "邮箱", "微信号", "手机号"],
        }

        # 记录消息里出现过关键词的所有意图大类，供分歧检测判断"LLM 选的领域有没有任何字面证据"。
        groups = {
            _group_of(cat)
            for patterns in (specific_patterns, generic_patterns)
            for cat, kws in patterns.items()
            if any(_keyword_in(kw, msg) for kw in kws)
        }

        best_cat, best_score = self._best_pattern_match(msg, specific_patterns)
        if best_cat == IntentCategory.OTHER:
            best_cat, best_score = self._best_pattern_match(msg, generic_patterns)
        return {"intent": best_cat, "confidence": best_score, "groups": groups}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        """加权投票。返回最终意图、融合置信度和各路来源得分。"""
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(emb.get("confidence", 0.0) or 0.0),
            "pattern": float(pat.get("confidence", 0.0) or 0.0),
        }
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"], source_scores["embedding"], source_scores
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"], source_scores["pattern"], source_scores
            return IntentCategory.OTHER, 0.0, source_scores

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        # 第一层：按意图大类汇总。"费用"和"退款"属于同一领域，分数应该合在一起和其他领域比，
        # 否则同领域的票被拆散，可能输给另一个单独得分更高的意图。
        group_scores: Dict[IntentCategory, float] = {}
        for cat, s in scores.items():
            g = _group_of(cat)
            group_scores[g] = group_scores.get(g, 0.0) + s
        best_group = max(group_scores, key=group_scores.get)  # type: ignore
        confidence = min(1.0, group_scores[best_group])

        # 分歧检测：向量和关键词两路确定性信号一致指向另一个领域时，
        # 即便 LLM 很自信也不直接采信，降为低置信 OTHER，交给澄清追问。
        min_emb = _CONFLICT_MIN_EMB_BGE if self._semantic not in (None, _UNSET) else _CONFLICT_MIN_EMB_NGRAM
        if self._deterministic_conflict(emb, pat, best_group, min_emb):
            source_scores["conflict"] = 1.0
            return IntentCategory.OTHER, min(confidence, self.threshold - 0.05), source_scores

        # 第二层：在胜出的大类内部选细分意图。
        members = {cat: s for cat, s in scores.items() if _group_of(cat) == best_group}
        best = max(members, key=members.get)  # type: ignore
        pat_intent = pat.get("intent", IntentCategory.OTHER)
        pat_conf = float(pat.get("confidence", 0.0) or 0.0)
        # 关键词细化只允许在同一大类内进行，不能把"留学咨询"改写成"退款"这种跨领域结果。
        if (
            best in _GENERIC_INTENTS
            and pat_intent in _SPECIFIC_INTENTS
            and _group_of(pat_intent) == best_group
            and pat_conf >= 0.5
            and confidence < 0.8
        ):
            source_scores["refined_by_pattern"] = pat_conf
            return pat_intent, max(confidence, pat_conf), source_scores

        if confidence < self.threshold:
            return IntentCategory.OTHER, confidence, source_scores
        return best, confidence, source_scores

    @staticmethod
    def _deterministic_conflict(
        emb: Dict, pat: Dict, chosen_group: IntentCategory, min_emb: float = _CONFLICT_MIN_EMB_NGRAM,
    ) -> bool:
        """向量与关键词是否一致指向另一个领域、两路信号都足够强，且消息里没有 LLM 所选领域的任何关键词。"""
        emb_intent = emb.get("intent", IntentCategory.OTHER)
        pat_intent = pat.get("intent", IntentCategory.OTHER)
        if IntentCategory.OTHER in (emb_intent, pat_intent):
            return False
        det_group = _group_of(emb_intent)
        if det_group != _group_of(pat_intent) or det_group == chosen_group:
            return False
        # 问候、请求、查询这类只描述语气、不指向具体业务，证据太弱，不能拿来否决 LLM 的业务判断。
        # （基准集上唯一一次误触发就是"改成用邮件联系我"被两路判成 request，否决了正确的 account。）
        if det_group in _PRAGMATIC_INTENTS:
            return False
        # 消息里也出现了 LLM 所选领域的关键词，说明可能是复合问题，不算分歧。
        if chosen_group in pat.get("groups", ()):
            return False
        return (
            float(emb.get("confidence", 0.0) or 0.0) >= min_emb
            and float(pat.get("confidence", 0.0) or 0.0) >= _CONFLICT_MIN_PAT
        )

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用规则提取高价值实体，避免每次识别都额外调用 LLM。"""
        message = self._clean_text(message)
        from business.catalog import match_service_mentions

        return {
            "contract_id": self._unique(re.findall(r"(?:合同|协议|订单)号?\s*[:：#]?\s*([A-Za-z]{1,4}-?\d{4,12})", message, re.I)),
            "date": self._unique(re.findall(r"(今天|明天|昨天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message)),
            "amount": self._unique(re.findall(r"((?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|块|rmb|cny|欧元|eur|€))", message, re.I)),
            "country": self._unique(
                re.findall(r"(瑞典|Sweden|德国|Germany|荷兰|Netherlands|芬兰|Finland|丹麦|Denmark)", message, re.I)
            ),
            "intake": self._unique(re.findall(r"(20\d{2}\s*(?:年)?\s*(?:秋|春|fall|autumn|spring)(?:季|季入学|入学)?)", message, re.I)),
            "test_score": self._unique(re.findall(r"((?:雅思|ielts|托福|toefl)\s*(?:总分)?\s*\d{1,3}(?:\.\d)?)", message, re.I)),
            "service": match_service_mentions(message),
        }

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = None
        semantic = self._semantic_embedder()
        if semantic is not None and hasattr(semantic, "embed_queries"):
            # 远端 API：一次批量编码全部模板，而不是逐条请求
            try:
                vecs = await asyncio.to_thread(semantic.embed_queries, all_texts)
            except Exception as ex:
                logger.warning(f"批量编码意图模板失败，逐条编码: {ex}")
        if vecs is None:
            vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """
        生成文本向量。

        优先使用和知识库共享的本地中文向量模型（bge-small-zh，约 1ms/条）；
        模型不可用时尝试远端 embeddings.create，再退化为字符 n-gram 哈希向量，
        保证 Embedding 缺失不会导致三路融合中断。
        基准测试（380 条）上，仅向量这一路的准确率：字符 n-gram 40.0%，bge 59.2%。
        """
        semantic = self._semantic_embedder()
        if semantic is not None:
            try:
                if hasattr(semantic, "embed_queries"):  # 远端 API：别阻塞事件循环
                    return await asyncio.to_thread(semantic.embed_query, text)
                return semantic.embed_query(text)
            except Exception as ex:
                logger.warning(f"中文向量模型编码失败，使用字符 n-gram 兜底: {ex}")

        embeddings = getattr(self.client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"远端 Embedding 失败，使用本地向量兜底: {ex}")

        return self._local_embedding(text)

    def _semantic_embedder(self):
        """懒加载共享的中文向量模型；GOEUROOPS_INTENT_EMBEDDING=ngram 时只用字符 n-gram。"""
        if self._semantic is _UNSET:
            self._semantic = None
            if os.getenv("GOEUROOPS_INTENT_EMBEDDING", "bge").strip().lower() != "ngram":
                from mcp.embeddings import get_shared_embedding_function

                self._semantic = get_shared_embedding_function()
        return self._semantic

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量（实现见 core/text_embedding.py）。"""
        return hashed_ngram_embedding(text, dims)

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        payload = {"message": self._clean_text(message)[:200]}
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", ""))[:160],
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if _keyword_in(kw, message))
            if not hits:
                continue
            # 单个明确业务关键词就给可用置信度；多个关键词命中时提高置信度。
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _intent_group(intent: IntentCategory) -> str:
        return _INTENT_GROUPS.get(intent, intent).value

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
