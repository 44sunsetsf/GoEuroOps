"""业务关键词表：意图识别的关键词规则和路由打分的领域关键词放在一处。

两份词表用途不同，所以不强行合成一份：
  - INTENT_KEYWORDS_*：意图识别第三路（关键词规则），按 19 个意图分组，决定“这句话是什么意图”；
  - ROUTING_KEYWORDS_*：路由打分，按 3 个领域（咨询 / 费用 / 前台）分组，决定“主 Agent 和协作 Agent 是谁”。
新增业务词时两边都要看一眼：tests/test_domain_terms.py 把两份词表现有的差异固定下来，
只改了其中一份时测试会失败，提醒你决定另一份要不要跟着改。
Skill 自己的 keywords 写在各自的 SKILL.md 里，属于 Skill 的配置，不放这里。
"""
from __future__ import annotations

from typing import Dict, List

from core.intent_categories import IntentCategory

# 细粒度业务意图的关键词：命中就优先判细类
INTENT_KEYWORDS_SPECIFIC: Dict[IntentCategory, List[str]] = {
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

# 通用大类和语气类意图的关键词：细类都没命中时才用
INTENT_KEYWORDS_GENERIC: Dict[IntentCategory, List[str]] = {
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

# 领域关键词：只用于主/辅 Agent 打分和复合问题检测，不直接决定路由
ROUTING_KEYWORDS_CONSULTING = ["留学", "申请", "硕士", "研究生", "选校", "文书", "雅思", "托福", "aps", "报价", "多少钱",
                               "套餐", "陪跑", "瑞典", "德国", "荷兰", "芬兰", "丹麦", "北欧"]
# 复合问题检测只看"问留学本身"的词：问退款时顺口提到"全程陪跑"不算咨询问题
ROUTING_KEYWORDS_CONSULTING_COLLAB = ["留学", "申请", "硕士", "研究生", "选校", "雅思", "托福", "aps",
                                      "瑞典", "德国", "荷兰", "芬兰", "丹麦", "北欧"]
ROUTING_KEYWORDS_BILLING = ["退款", "退钱", "定金", "尾款", "发票", "付款", "多付", "refund", "invoice"]
ROUTING_KEYWORDS_GENERAL = ["进度", "第几轮", "你们是", "工作室", "联系方式", "帮助"]
