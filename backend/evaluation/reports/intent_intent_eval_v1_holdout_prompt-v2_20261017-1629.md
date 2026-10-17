# 意图识别基准：intent_eval_v1

- 样本 190 条，19 类；模型 `deepseek-v4-flash`；向量路 `bge`；运行于 2026-10-17T16:29:55，耗时 32.6 秒
- LLM 调用失败 0 次；LLM 单次延迟 p50 990 ms，p95 1219 ms
- 分歧检测触发 0 次；子集 `holdout`；prompt-v2
- 误转人工 0 条（0.0%，不该转的样本里被判成转人工/升级）；漏转人工 1 条（5.0%）

## 总体与消融

| 识别器 | 准确率 | Macro-F1 | 领域准确率（决定路由） |
|---|---|---|---|
| 三路融合（v3.1，线上） | 96.8% | 0.968 | 97.4% |
| 三路融合（v3.0，修复前） | 95.8% | 0.957 | 96.3% |
| 仅 LLM | 96.8% | 0.968 | 97.4% |
| 仅向量模板 | 57.9% | 0.538 | 64.2% |
| 仅关键词 | 36.3% | 0.410 | 38.4% |
| LLM 故障时的降级 | 57.9% | 0.538 | 64.2% |

## 各类别（三路融合）

| 意图 | Precision | Recall | F1 | 样本 |
|---|---|---|---|---|
| billing | 1.00 | 0.80 | 0.89 | 10 |
| query | 0.90 | 0.90 | 0.90 | 10 |
| greeting | 0.83 | 1.00 | 0.91 | 10 |
| request | 1.00 | 0.90 | 0.95 | 10 |
| feedback | 1.00 | 0.90 | 0.95 | 10 |
| human_handoff | 1.00 | 0.90 | 0.95 | 10 |
| account | 0.91 | 1.00 | 0.95 | 10 |
| invoice | 0.91 | 1.00 | 0.95 | 10 |
| service_inquiry | 0.91 | 1.00 | 0.95 | 10 |
| complaint | 1.00 | 1.00 | 1.00 | 10 |
| escalation | 1.00 | 1.00 | 1.00 | 10 |
| study_consult | 1.00 | 1.00 | 1.00 | 10 |
| service_progress | 1.00 | 1.00 | 1.00 | 10 |
| booking | 1.00 | 1.00 | 1.00 | 10 |
| refund | 1.00 | 1.00 | 1.00 | 10 |
| payment_issue | 1.00 | 1.00 | 1.00 | 10 |
| data_privacy | 1.00 | 1.00 | 1.00 | 10 |
| application_process | 1.00 | 1.00 | 1.00 | 10 |
| other | 1.00 | 1.00 | 1.00 | 10 |

## 按难点标签

| 标签 | 样本 | 准确率 |
|---|---|---|
| boundary | 28 | 85.7% |
| colloquial | 12 | 100.0% |
| context | 5 | 100.0% |
| mixed | 12 | 100.0% |
| plain | 131 | 98.5% |
| typo | 2 | 100.0% |

## 最常见的混淆

| 标注 | 预测 | 次数 |
|---|---|---|
| query | greeting | 1 |
| request | account | 1 |
| billing | service_inquiry | 1 |
| billing | invoice | 1 |
| feedback | greeting | 1 |
| human_handoff | query | 1 |

## v3.1 修复的影响

- 修复前错、修复后对：3 条 ['你是真人还是机器人', '我要投诉你们的退款处理', '付款截止是哪天']
- 修复前对、修复后错：1 条 ['有真人吗']

## 错例

| 文本 | 标注 | 预测 | LLM | 置信度 |
|---|---|---|---|---|
| 怎么称呼你 | query | greeting | greeting | 0.59 |
| 帮我把目标国家改成只看荷兰和德国 | request | account | account | 0.59 |
| 收费是按项目还是按年 | billing | service_inquiry | service_inquiry | 0.68 |
| 付款后会有收据吗 | billing | invoice | invoice | 0.77 |
| 好的明白了，谢谢 | feedback | greeting | greeting | 0.59 |
| 有真人吗 | human_handoff | query | query | 0.59 |
