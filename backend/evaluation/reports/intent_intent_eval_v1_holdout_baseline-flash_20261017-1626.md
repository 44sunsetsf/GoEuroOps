# 意图识别基准：intent_eval_v1

- 样本 190 条，19 类；模型 `deepseek-v4-flash`；向量路 `bge`；运行于 2026-10-17T16:26:59，耗时 33.9 秒
- LLM 调用失败 0 次；LLM 单次延迟 p50 1059 ms，p95 1388 ms
- 分歧检测触发 0 次；子集 `holdout`；baseline-flash
- 误转人工 3 条（1.8%，不该转的样本里被判成转人工/升级）；漏转人工 1 条（5.0%）

## 总体与消融

| 识别器 | 准确率 | Macro-F1 | 领域准确率（决定路由） |
|---|---|---|---|
| 三路融合（v3.1，线上） | 90.0% | 0.896 | 92.1% |
| 三路融合（v3.0，修复前） | 88.9% | 0.885 | 91.0% |
| 仅 LLM | 90.0% | 0.896 | 92.1% |
| 仅向量模板 | 57.9% | 0.538 | 64.2% |
| 仅关键词 | 36.3% | 0.410 | 38.4% |
| LLM 故障时的降级 | 57.9% | 0.538 | 64.2% |

## 各类别（三路融合）

| 意图 | Precision | Recall | F1 | 样本 |
|---|---|---|---|---|
| query | 1.00 | 0.50 | 0.67 | 10 |
| request | 1.00 | 0.50 | 0.67 | 10 |
| human_handoff | 0.60 | 0.90 | 0.72 | 10 |
| service_inquiry | 0.67 | 1.00 | 0.80 | 10 |
| escalation | 1.00 | 0.70 | 0.82 | 10 |
| other | 0.82 | 0.90 | 0.86 | 10 |
| billing | 1.00 | 0.80 | 0.89 | 10 |
| greeting | 0.90 | 0.90 | 0.90 | 10 |
| feedback | 1.00 | 0.90 | 0.95 | 10 |
| service_progress | 0.91 | 1.00 | 0.95 | 10 |
| booking | 0.91 | 1.00 | 0.95 | 10 |
| invoice | 0.91 | 1.00 | 0.95 | 10 |
| payment_issue | 0.91 | 1.00 | 0.95 | 10 |
| application_process | 0.91 | 1.00 | 0.95 | 10 |
| complaint | 1.00 | 1.00 | 1.00 | 10 |
| study_consult | 1.00 | 1.00 | 1.00 | 10 |
| account | 1.00 | 1.00 | 1.00 | 10 |
| refund | 1.00 | 1.00 | 1.00 | 10 |
| data_privacy | 1.00 | 1.00 | 1.00 | 10 |

## 按难点标签

| 标签 | 样本 | 准确率 |
|---|---|---|
| boundary | 28 | 67.9% |
| colloquial | 12 | 91.7% |
| context | 5 | 100.0% |
| mixed | 12 | 100.0% |
| plain | 131 | 93.1% |
| typo | 2 | 100.0% |

## 最常见的混淆

| 标注 | 预测 | 次数 |
|---|---|---|
| query | service_inquiry | 3 |
| escalation | human_handoff | 3 |
| query | greeting | 1 |
| query | human_handoff | 1 |
| request | payment_issue | 1 |
| request | service_progress | 1 |
| request | human_handoff | 1 |
| request | other | 1 |
| request | application_process | 1 |
| greeting | human_handoff | 1 |
| billing | service_inquiry | 1 |
| billing | invoice | 1 |

## v3.1 修复的影响

- 修复前错、修复后对：2 条 ['我要投诉你们的退款处理', '付款截止是哪天']
- 修复前对、修复后错：0 条 []

## 错例

| 文本 | 标注 | 预测 | LLM | 置信度 |
|---|---|---|---|---|
| 怎么称呼你 | query | greeting | greeting | 0.59 |
| 顾问是全职的吗还是兼职 | query | service_inquiry | service_inquiry | 0.59 |
| 你是真人还是机器人 | query | human_handoff | human_handoff | 0.67 |
| 你们顾问都是哪个学校毕业的 | query | service_inquiry | service_inquiry | 0.59 |
| 你们只做计算机方向吗 | query | service_inquiry | service_inquiry | 0.78 |
| 麻烦发一下付款确认截图给我 | request | payment_issue | payment_issue | 0.57 |
| 请把文书的批注版也发我 | request | service_progress | service_progress | 0.59 |
| 帮我把我的背景资料转给新顾问 | request | human_handoff | human_handoff | 0.73 |
| 帮我记一下我下个月雅思考试 | request | other | other | 0.50 |
| 能不能把申请时间表发我邮箱 | request | application_process | application_process | 0.59 |
| 有人在线吗 | greeting | human_handoff | human_handoff | 0.50 |
| 这个问题我要直接和创始人谈 | escalation | human_handoff | human_handoff | 0.87 |
| 你们创始人的联系方式给我 | escalation | human_handoff | human_handoff | 0.71 |
| 我需要你们给一个正式的书面答复 | escalation | human_handoff | human_handoff | 0.63 |
| 收费是按项目还是按年 | billing | service_inquiry | service_inquiry | 0.71 |
| 付款后会有收据吗 | billing | invoice | invoice | 0.80 |
| 好的明白了，谢谢 | feedback | other | other | 0.63 |
| 我的情况比较复杂，想和顾问一对一聊 | human_handoff | booking | booking | 0.59 |
| 日本留学你们做吗 | other | service_inquiry | service_inquiry | 0.63 |
