"""意图类别枚举。

单独成一个模块，业务词表（business/domain_terms.py）和意图识别器都要用它，放在识别器里会循环导入。
"""
from enum import Enum


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
