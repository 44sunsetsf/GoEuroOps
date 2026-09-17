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

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"

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

        # 使用服务端时不传 embedding_function，让服务端处理
        # 本地模式时也不传，使用 ChromaDB 默认的（会触发模型下载）
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "GoEuroOps RAG 知识库"},
        )

        # 如果知识库为空，导入默认文档
        if self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "...", "domain": "..."}, ...]
        长文档会自动切片（每片 500 字）。

        domain 标注文档归属哪个 Agent（如 "consulting"/"billing"/"general"），
        检索时按调用方 Agent 的 domain 过滤，避免不同业务线的知识互相串场；
        不填时默认为 "shared"，对所有 Agent 可见。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            domain  = str(doc.get("domain") or "shared")
            chunks  = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({"title": title, "chunk_index": i, "total_chunks": len(chunks), "domain": domain})

        if ids:
            # ChromaDB 会自动生成 Embedding
            self._collection.add(ids=ids, documents=docs, metadatas=metas)
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
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                })

        return items

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

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（客服场景常见问题）。"""
        default_docs = [
            {
                "title": "退款政策",
                "domain": "billing",
                "content": (
                    "退款政策说明。"
                    "用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "domain": "general",
                "content": (
                    "订单查询指南。"
                    "用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "账户安全",
                "domain": "billing",
                "content": (
                    "账户安全说明。"
                    "建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "工作室服务介绍（示例参考信息）",
                "domain": "consulting",
                "content": (
                    "工作室服务介绍，示例参考信息，请以工作室顾问最新说明为准。"
                    "我们是由几位北欧留学生创立的留学咨询工作室，专注于英语授课计算机硕士方向的申请支持。"
                    "核心服务分两类：一是选校与项目定位咨询，帮助梳理背景、匹配候选院校和项目；"
                    "二是文书与个人陈述写作辅导，由顾问一对一打磨内容结构和表达。"
                    "这两项服务均由工作室创始人亲自提供，机器人助手仅负责介绍服务内容和解答公开知识问题，"
                    "不代为完成选校决策或文书写作。如需具体咨询，请通过对话留下联系方式，顾问会安排沟通时间。"
                ),
            },
            {
                "title": "瑞典CS硕士申请概览（示例参考信息）",
                "domain": "consulting",
                "content": (
                    "瑞典计算机硕士申请概览，示例参考信息，请以院校官网和工作室顾问为准。"
                    "瑞典多所大学提供英语授课的计算机科学/软件工程硕士项目，学制通常为2年（120学分）。"
                    "英语能力要求通常参考雅思或同等托福成绩，具体分数线因校因项目而异。"
                    "主申请季通常集中在每年1月中旬截止，对应次年秋季入学。"
                    "学费方面，欧盟/欧洲经济区学生通常享受免学费待遇，非欧盟学生一般需要缴纳学费，具体以院校公告为准。"
                ),
            },
            {
                "title": "德国CS硕士申请概览（示例参考信息）",
                "domain": "consulting",
                "content": (
                    "德国计算机硕士申请概览，示例参考信息，请以院校官网和工作室顾问为准。"
                    "德国有不少高校提供英语授课的计算机科学硕士项目，学制多为2年，也存在部分1.5年项目。"
                    "语言要求以英语为主，部分项目会要求基础德语；英语能力通常参考雅思或同等托福成绩。"
                    "多数联邦州公立大学学费较低甚至免学费，仅收取每学期少量注册/杂费，具体因州因校而异。"
                    "申请截止时间因校而异，建议提前规划，冬季学期入学申请通常在当年较早时间截止。"
                ),
            },
            {
                "title": "荷兰CS硕士申请概览（示例参考信息）",
                "domain": "consulting",
                "content": (
                    "荷兰计算机硕士申请概览，示例参考信息，请以院校官网和工作室顾问为准。"
                    "荷兰多所研究型大学提供英语授课的计算机科学相关硕士项目，学制常见为1年或2年，因校因项目而异。"
                    "语言要求通常参考雅思或同等托福成绩，具体门槛以院校公告为准。"
                    "申请截止时间因校而异，部分热门项目截止较早，建议提前准备材料。"
                    "学费方面欧盟学生通常较低，非欧盟学生学费相对较高，具体以院校最新公告为准。"
                ),
            },
            {
                "title": "芬兰CS硕士申请概览（示例参考信息）",
                "domain": "consulting",
                "content": (
                    "芬兰计算机硕士申请概览，示例参考信息，请以院校官网和工作室顾问为准。"
                    "芬兰高校普遍提供英语授课的计算机科学硕士项目，学制通常为2年。"
                    "语言要求通常参考雅思或同等托福成绩，部分项目也接受其他英语能力证明。"
                    "主申请季通常在每年1月截止，对应次年秋季入学。"
                    "学费方面，欧盟/欧洲经济区学生通常免学费，非欧盟学生一般需缴纳学费，部分院校设有奖学金，具体以官网为准。"
                ),
            },
            {
                "title": "丹麦CS硕士申请概览（示例参考信息）",
                "domain": "consulting",
                "content": (
                    "丹麦计算机硕士申请概览，示例参考信息，请以院校官网和工作室顾问为准。"
                    "丹麦多所大学提供英语授课的计算机科学硕士项目，学制通常为2年。"
                    "语言要求通常参考雅思或同等托福成绩，具体门槛以院校公告为准。"
                    "非欧盟申请者的截止日期通常较早（常见于当年较早月份），欧盟申请者截止相对较晚，建议提前确认具体日期。"
                    "学费方面欧盟/欧洲经济区学生通常免学费，非欧盟学生一般需缴纳学费，具体以院校最新公告为准。"
                ),
            },
            {
                "title": "申请材料与时间线概览（示例参考信息）",
                "domain": "consulting",
                "content": (
                    "留学申请材料与时间线概览，示例参考信息，请以院校官网和工作室顾问为准。"
                    "常见申请材料包括：本科成绩单、学位证明、英语能力证明（雅思/托福等）、个人陈述、推荐信、简历，部分项目还要求作品集或GRE成绩。"
                    "建议至少提前6-12个月开始准备，优先确认目标院校的语言要求和申请截止日期，这两项因校而异且可能调整。"
                    "个人陈述和推荐信通常需要较长打磨时间，建议尽早开始草拟。"
                    "具体材料清单、格式要求和截止日期请以各院校官网最新公告为准，工作室顾问可协助核实和规划时间线。"
                ),
            },
            {
                "title": "会员与积分",
                "domain": "general",
                "content": (
                    "会员积分规则。"
                    "每消费 1 元累积 1 积分。"
                    "积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。"
                    "积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "domain": "general",
                "content": (
                    "配送服务说明。"
                    "标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。"
                    "同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。"
                    "配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
