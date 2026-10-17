# 意图识别基准：intent_eval_v1

- 样本 190 条，19 类；模型 `deepseek-v4-flash`；向量路 `bge`；运行于 2026-10-17T16:25:50，耗时 33.8 秒
- LLM 调用失败 0 次；LLM 单次延迟 p50 1026 ms，p95 1270 ms
- 分歧检测触发 0 次；子集 `dev`；baseline-flash
- 误转人工 1 条（0.6%，不该转的样本里被判成转人工/升级）；漏转人工 0 条（0.0%）

## 总体与消融

| 识别器 | 准确率 | Macro-F1 | 领域准确率（决定路由） |
|---|---|---|---|
| 三路融合（v3.1，线上） | 91.6% | 0.912 | 94.7% |
| 三路融合（v3.0，修复前） | 91.6% | 0.912 | 94.7% |
| 仅 LLM | 91.6% | 0.912 | 94.7% |
| 仅向量模板 | 60.5% | 0.583 | 67.9% |
| 仅关键词 | 41.0% | 0.482 | 44.2% |
| LLM 故障时的降级 | 60.5% | 0.583 | 67.9% |

## 各类别（三路融合）

| 意图 | Precision | Recall | F1 | 样本 |
|---|---|---|---|---|
| escalation | 1.00 | 0.40 | 0.57 | 10 |
| human_handoff | 0.59 | 1.00 | 0.74 | 10 |
| query | 1.00 | 0.60 | 0.75 | 10 |
| request | 1.00 | 0.60 | 0.75 | 10 |
| service_inquiry | 0.67 | 1.00 | 0.80 | 10 |
| other | 0.82 | 0.90 | 0.86 | 10 |
| billing | 1.00 | 0.90 | 0.95 | 10 |
| study_consult | 0.91 | 1.00 | 0.95 | 10 |
| service_progress | 0.91 | 1.00 | 0.95 | 10 |
| complaint | 1.00 | 1.00 | 1.00 | 10 |
| greeting | 1.00 | 1.00 | 1.00 | 10 |
| account | 1.00 | 1.00 | 1.00 | 10 |
| feedback | 1.00 | 1.00 | 1.00 | 10 |
| booking | 1.00 | 1.00 | 1.00 | 10 |
| refund | 1.00 | 1.00 | 1.00 | 10 |
| invoice | 1.00 | 1.00 | 1.00 | 10 |
| payment_issue | 1.00 | 1.00 | 1.00 | 10 |
| data_privacy | 1.00 | 1.00 | 1.00 | 10 |
| application_process | 1.00 | 1.00 | 1.00 | 10 |

## 按难点标签

| 标签 | 样本 | 准确率 |
|---|---|---|
| boundary | 20 | 75.0% |
| colloquial | 13 | 92.3% |
| context | 4 | 100.0% |
| long | 1 | 100.0% |
| mixed | 12 | 91.7% |
| plain | 140 | 93.6% |

## 最常见的混淆

| 标注 | 预测 | 次数 |
|---|---|---|
| escalation | human_handoff | 6 |
| query | service_inquiry | 3 |
| query | other | 1 |
| request | study_consult | 1 |
| request | other | 1 |
| request | service_progress | 1 |
| request | human_handoff | 1 |
| billing | service_inquiry | 1 |
| other | service_inquiry | 1 |

## v3.1 修复的影响

- 修复前错、修复后对：0 条 []
- 修复前对、修复后错：0 条 []

## 错例

| 文本 | 标注 | 预测 | LLM | 置信度 |
|---|---|---|---|---|
| 可以用英文沟通吗 | query | other | other | 0.49 |
| 你们和那些大中介有什么区别 | query | service_inquiry | service_inquiry | 0.59 |
| 你们一般用什么软件开会 | query | service_inquiry | service_inquiry | 0.59 |
| 你们一年带多少学生 | query | service_inquiry | service_inquiry | 0.59 |
| 能给我开个项目对比表吗 | request | study_consult | study_consult | 0.59 |
| 需要你们出一个服务证明 | request | other | service_progress | 0.39 |
| please resend the timeline file | request | service_progress | service_progress | 0.50 |
| 帮忙提醒一下顾问看我的邮件 | request | human_handoff | human_handoff | 0.76 |
| 我要找你们老板 | escalation | human_handoff | human_handoff | 0.77 |
| 我不想跟客服说了，找负责人 | escalation | human_handoff | human_handoff | 0.71 |
| 你们主管是谁，我要跟他说 | escalation | human_handoff | human_handoff | 0.75 |
| 把你们负责人叫来 | escalation | human_handoff | human_handoff | 0.85 |
| 找个能做主的人来 | escalation | human_handoff | human_handoff | 0.80 |
| 把这件事交给你们的管理者 | escalation | human_handoff | human_handoff | 0.76 |
| 费用包含申请费吗 | billing | service_inquiry | service_inquiry | 0.59 |
| MBA申请你们接吗 | other | service_inquiry | service_inquiry | 0.76 |
