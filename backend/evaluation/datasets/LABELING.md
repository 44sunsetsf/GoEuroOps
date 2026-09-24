# 意图评测集标注规则（intent_eval_v1）

- 规模：19 类 × 20 条 = 380 条，类别均衡。样本定义在 `build_intent_v1.py`，运行它生成 `intent_eval_v1.jsonl`。
- 来源：按工作室真实咨询场景编写，覆盖口语、错别字、中英混杂、依赖上文和边界混淆五类难点（`tags` 字段）。
- 去重：跑分脚本会检查样本是否和识别器内置的 few-shot / 向量模板重复，重复就拒绝运行，防止分数虚高。
- 冻结：v1 发布后不根据模型预测改标签。标签有争议的样本记录在本文件末尾，统一在 v2 里处理。

## 判定顺序

1. 指向**具体业务对象**的，判细分意图：退款、发票、付款异常、预约、服务进度、隐私、申请流程、服务询价、转人工。
2. 只知道领域、说不出具体事的，判大类：费用（billing）、留学咨询（study_consult）、个人资料（account）。
3. 不指向任何业务的，按语用判：问候、正面反馈、投诉、请求、一般查询。
4. 工作室业务范围以外的（其他国家、非 CS 专业、闲聊、写代码），判 other。

## 容易混淆的边界

| 边界 | 规则 | 例子 |
|---|---|---|
| query / service_inquiry | 问工作室本身（团队、资质、地点）是 query；问服务内容、价格、优惠是 service_inquiry | "你们一年带多少学生" → query；"全程陪跑包含什么" → service_inquiry |
| escalation / human_handoff | 找负责人、正式投诉、要书面答复是 escalation；只是想和真人顾问说话是 human_handoff | "把你们负责人叫来" → escalation；"转真人" → human_handoff |
| complaint / escalation | 表达不满但没要求升级是 complaint；要求找负责人或正式投诉是 escalation | "又延期了无语" → complaint |
| billing / refund / payment_issue | 付款方式等泛问题是 billing；要退钱是 refund；付款失败、多付、未到账是 payment_issue | "能分期吗" → billing |
| request / 细分意图 | 请求操作且不属于任何细分业务时才是 request | "把会议纪要发我" → request；"帮我约咨询" → booking |
| other / service_inquiry | 问工作室**不提供**的服务（日本、MBA）判 other | "MBA 申请你们接吗" → other |

## 有争议、留到 v2 处理的样本

| 样本 | v1 标签 | 争议 |
|---|---|---|
| 日本留学你们做吗 / MBA申请你们接吗 | other | 也可理解为在问服务范围（service_inquiry） |
| 好的明白了，谢谢 | feedback | 更像对话结束语，可能应单列或判 greeting |
| 你是真人还是机器人 | query | 可能隐含转人工诉求 |
| 麻烦把推荐信模板发我 | request | 推荐信属于申请材料，也可判 application_process |

注意：escalation 和 human_handoff 都路由到转顾问 Agent，两者混淆不影响路由。报告里的"领域准确率"按大类计分，反映的就是路由层面的准确率。
