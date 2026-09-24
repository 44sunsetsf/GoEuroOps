# 意图识别基准：intent_eval_v1

- 样本 380 条，19 类；模型 `deepseek-v4-pro`；向量路 `bge`；运行于 2026-09-24T11:12:05，耗时 92.1 秒
- LLM 调用失败 0 次；LLM 单次延迟 p50 1420 ms，p95 1746 ms
- 分歧检测触发 0 次

## 总体与消融

| 识别器 | 准确率 | Macro-F1 | 领域准确率（决定路由） |
|---|---|---|---|
| 三路融合（v3.1，线上） | 91.6% | 0.914 | 94.0% |
| 三路融合（v3.0，修复前） | 91.3% | 0.911 | 93.7% |
| 仅 LLM | 91.6% | 0.914 | 94.0% |
| 仅向量模板 | 59.2% | 0.565 | 66.0% |
| 仅关键词 | 38.7% | 0.449 | 41.3% |
| LLM 故障时的降级 | 59.2% | 0.565 | 66.0% |

## 各类别（三路融合）

| 意图 | Precision | Recall | F1 | 样本 |
|---|---|---|---|---|
| query | 0.91 | 0.50 | 0.65 | 20 |
| escalation | 1.00 | 0.55 | 0.71 | 20 |
| service_inquiry | 0.62 | 1.00 | 0.77 | 20 |
| human_handoff | 0.67 | 1.00 | 0.80 | 20 |
| request | 0.94 | 0.85 | 0.89 | 20 |
| billing | 0.94 | 0.85 | 0.89 | 20 |
| greeting | 0.90 | 0.95 | 0.93 | 20 |
| application_process | 0.87 | 1.00 | 0.93 | 20 |
| other | 1.00 | 0.90 | 0.95 | 20 |
| feedback | 0.95 | 0.95 | 0.95 | 20 |
| complaint | 1.00 | 0.95 | 0.97 | 20 |
| account | 1.00 | 0.95 | 0.97 | 20 |
| service_progress | 1.00 | 0.95 | 0.97 | 20 |
| invoice | 0.95 | 1.00 | 0.98 | 20 |
| study_consult | 1.00 | 1.00 | 1.00 | 20 |
| booking | 1.00 | 1.00 | 1.00 | 20 |
| refund | 1.00 | 1.00 | 1.00 | 20 |
| payment_issue | 1.00 | 1.00 | 1.00 | 20 |
| data_privacy | 1.00 | 1.00 | 1.00 | 20 |

## 按难点标签

| 标签 | 样本 | 准确率 |
|---|---|---|
| boundary | 48 | 72.9% |
| colloquial | 25 | 88.0% |
| context | 9 | 100.0% |
| long | 1 | 100.0% |
| mixed | 24 | 100.0% |
| plain | 271 | 94.1% |
| typo | 2 | 100.0% |

## 最常见的混淆

| 标注 | 预测 | 次数 |
|---|---|---|
| query | service_inquiry | 8 |
| escalation | human_handoff | 8 |
| request | application_process | 2 |
| billing | service_inquiry | 2 |
| other | service_inquiry | 2 |
| query | greeting | 1 |
| query | human_handoff | 1 |
| complaint | feedback | 1 |
| request | billing | 1 |
| greeting | human_handoff | 1 |
| escalation | query | 1 |
| billing | invoice | 1 |

## v3.1 修复的影响

- 修复前错、修复后对：1 条 ['付款截止是哪天']
- 修复前对、修复后错：0 条 []

## 错例

| 文本 | 标注 | 预测 | LLM | 置信度 |
|---|---|---|---|---|
| 顾问是全职的吗还是兼职 | query | service_inquiry | service_inquiry | 0.64 |
| 你们平时几点上班 | query | service_inquiry | service_inquiry | 0.59 |
| 你们顾问都是哪个学校毕业的 | query | service_inquiry | service_inquiry | 0.64 |
| 你们和那些大中介有什么区别 | query | service_inquiry | service_inquiry | 0.64 |
| 可以用英文沟通吗 | query | service_inquiry | service_inquiry | 0.69 |
| 你们只做计算机方向吗 | query | service_inquiry | service_inquiry | 0.82 |
| 你们一年带多少学生 | query | service_inquiry | service_inquiry | 0.64 |
| 你们是正规公司吗 | query | service_inquiry | service_inquiry | 0.63 |
| 怎么称呼你 | query | greeting | greeting | 0.63 |
| 你是真人还是机器人 | query | human_handoff | human_handoff | 0.83 |
| 你们这个服务真的一般 | complaint | feedback | feedback | 0.63 |
| 麻烦发一下付款确认截图给我 | request | billing | billing | 0.83 |
| 帮我记一下我下个月雅思考试 | request | application_process | application_process | 0.80 |
| 麻烦把推荐信模板发我 | request | application_process | application_process | 0.63 |
| 有人在线吗 | greeting | human_handoff | human_handoff | 0.63 |
| 把你们负责人叫来 | escalation | human_handoff | human_handoff | 0.89 |
| 这个问题我要直接和创始人谈 | escalation | human_handoff | human_handoff | 0.92 |
| 你们主管是谁，我要跟他说 | escalation | human_handoff | human_handoff | 0.77 |
| 找个能做主的人来 | escalation | human_handoff | human_handoff | 0.81 |
| 请上报给你们合伙人 | escalation | human_handoff | human_handoff | 0.78 |
| 我需要你们给一个正式的书面答复 | escalation | human_handoff | human_handoff | 0.70 |
| 你们创始人的联系方式给我 | escalation | query | query | 0.63 |
| 我不想跟客服说了，找负责人 | escalation | human_handoff | human_handoff | 0.73 |
| 把这件事交给你们的管理者 | escalation | human_handoff | human_handoff | 0.78 |
| 付款后会有收据吗 | billing | invoice | invoice | 0.82 |
| 费用包含申请费吗 | billing | service_inquiry | service_inquiry | 0.64 |
| 收费是按项目还是按年 | billing | service_inquiry | service_inquiry | 0.71 |
| 我的名字拼音写错了 | account | request | request | 0.63 |
| 好的明白了，谢谢 | feedback | greeting | greeting | 0.63 |
| 我的APS材料清单准备好了吗 | service_progress | application_process | application_process | 0.86 |
| 日本留学你们做吗 | other | service_inquiry | service_inquiry | 0.67 |
| MBA申请你们接吗 | other | service_inquiry | service_inquiry | 0.78 |
