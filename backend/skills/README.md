# GoEuroOps Skills 编写规范

Skill 是一份**可热加载的业务规范**：话术、处理流程、边界、升级条件。它不改代码，由运营（创始人）直接维护，命中后注入对应 Agent 的 system prompt。

设计参照 Anthropic Agent Skills 的**渐进式披露**：核心规则常驻，详细资料按需读取。和官方做法的区别是，本项目的 Agent 是固定的业务角色、对延迟敏感，所以"用不用某个 Skill"由确定性路由决定（意图 + 关键词 + 语义样例 + 会话粘性），不额外花一轮 LLM 调用让模型自己挑。

## 当前 Skills

| id | 适用 Agent | 模式 | 作用 |
|---|---|---|---|
| `studio_brand_voice` | 全部 | always | 品牌语气、时差、隐私与承诺底线（≤600 字） |
| `general_reception` | general | auto | 接待、分诊、售后进度、投诉首轮沟通 |
| `study_consulting` | consulting | auto | 五国 CS 硕士公开知识答疑、个性化请求转顾问 |
| `service_sales` | consulting | auto | 服务介绍、报价必须走工具、价格异议、线索登记 |
| `billing_support` | billing | auto | 定金尾款、退款估算、发票 |

## 目录结构

```text
skills/<skill_id>/
├── SKILL.md            # 必需：front matter + 核心规则（命中即注入，≤2500 字）
├── references/*.md     # 可选：详细资料（话术库、FAQ、案例），只列目录，模型按需读取
└── evals/cases.json    # 推荐：命中回归用例，CI 和评测面板都会跑
```

目录名就是 Skill 的 `id`。

## SKILL.md front matter

```yaml
---
name: 服务介绍与报价规范            # 必填，全局唯一，会出现在 prompt 里
description: 一句话说明用途          # 必填，也参与语义匹配
version: 1.0.0                     # 改规则时递增，响应和 trace 里会带上版本
owner: 创始人团队                   # 谁维护
updated_at: 2026-09-23
agents: [consulting]               # 适用 Agent：general / consulting / billing / escalation；不写=全部
intents: [service_inquiry, booking]  # 绑定意图（最强信号），取值见 core/intent_recognizer.py
keywords: [价格, 多少钱, 报价, 优惠]   # 触发关键词；YAML 列表或逗号分隔都行（中英文逗号均可）
examples:                          # 语义触发样例：覆盖关键词表之外的说法
  - 能不能便宜一点
  - 太贵了
mode: auto                         # auto=按信号命中；always=对适用 Agent 常驻
priority: 55                       # 同时命中时的注入顺序（大的在前），always 永远最前
enabled: true
---
```

加载时会做校验，**不合格的 Skill 会被跳过**并出现在 `GET /skills` 的 `errors` 里：缺 name/description、未知 agents/intents、`mode: auto` 却没有任何触发条件、name 重复。正文超过 3200 字会给出警告，注入时会被截断。

## 命中规则

对每个适用当前 Agent 的 Skill 打分，**≥ 0.35**（`GOEUROOPS_SKILL_MATCH_THRESHOLD`）即注入：

| 信号 | 分值 | 说明 |
|---|---|---|
| 意图绑定 | +0.50 | 本轮具体意图在 `intents` 里，单独就能命中 |
| 意图大类 | +0.25 | 只有意图大类在 `intents` 里（如 `service_inquiry` 属于 `study_consult` 组），单独不足以命中 |
| 关键词 | +0.35 起，每多一个 +0.05，最多 0.45 | 中文做子串匹配；纯英文关键词按单词边界匹配（`cs` 不会命中 `docs`） |
| 语义样例 | 最多 +0.40 | 用户消息和 `examples`/`description` 的字符 n-gram 相似度 |
| 会话粘性 | 补足到阈值 | 上一轮用户消息单独就能命中时，本轮追问（"那奖学金呢"）继承命中 |
| 常驻 | 1.0 | `mode: always` |

命中的 Skill 按 always → priority → 得分排序，总长度不超过 6000 字（`GOEUROOPS_SKILLS_MAX_PROMPT_CHARS`），超出的会被跳过并记录。

## references：按需读取的资料

- 命中的 Skill 只会把 references 的**文件名和标题**列进 prompt；
- 模型需要时调用 `read_skill_reference(skill, file)` 读取全文（单次最多 6000 字）；
- 只能读取**本轮已命中** Skill 的已登记文件，不接受路径（防止越权读取其他文件）。

适合放进 references 的：长话术库、FAQ、计算示例、回复模板。经常用到、必须遵守的规则留在 SKILL.md。

## evals/cases.json：命中回归用例

```json
[
  {"message": "全程陪跑多少钱", "agent": "consulting", "expect_hit": true},
  {"message": "我想约一下顾问", "agent": "consulting", "intent": "booking", "expect_hit": true},
  {"message": "瑞典的CS硕士一般读几年？", "agent": "consulting", "intent": "study_consult", "expect_hit": false},
  {"message": "那丹麦呢？", "agent": "consulting",
   "history": [{"role": "user", "content": "瑞典的计算机硕士学费多少"}], "expect_hit": true}
]
```

- 不写 `intent` 的用例专门验证关键词/语义能否兜住意图识别失误的情况；
- 一定要写几条 `expect_hit: false`，防止关键词加太宽导致误命中。

运行方式：

```bash
pytest tests/test_skill_loader.py            # 本地 / CI
curl http://localhost:8000/skills/evals       # 线上
```

前端「Skills」页的「运行命中回归」也是调这个接口；`/eval/run` 会把准确率作为 `skill_routing_accuracy` 一起报告。

## 调试与发布

```bash
# 某句话会命中哪些 Skill、各信号多少分（不传 intent 时现场做意图识别）
curl -X POST http://localhost:8000/skills/match \
  -H 'Content-Type: application/json' \
  -d '{"message": "想去北欧读计算机研究生", "agent_type": "consulting"}'

# 热加载（Docker 中 skills/ 是挂载的，改完直接生效，不用重建镜像）
curl -X POST http://localhost:8000/skills/reload

# 查看某个 Skill 的正文、references 和命中统计
curl http://localhost:8000/skills/service_sales
```

每次对话的响应（`/chat` 的 `skills_applied`、SSE `done` 事件、`/trace/tool/{id}`）都会带上本轮注入了哪些 Skill、版本和得分原因；Prometheus 指标 `goeuroops_skill_hits_total{skill,agent}` 记录命中次数。

**发布检查清单**

1. `version`、`updated_at` 已更新；
2. 新增的关键词/样例补了对应的 `evals` 用例（包括不该命中的）；
3. `pytest tests/test_skill_loader.py` 通过；
4. `POST /skills/reload` 后 `GET /skills` 的 `errors` 为空；
5. 在「Skills」页用命中测试试 2–3 句真实用户说法。

## 编写原则

- 重要规则写在前面；正文控制在 2500 字以内，细节移到 references。
- 一个 Skill 只管一类职责。品牌语气这类所有 Agent 都要遵守的，放 `studio_brand_voice`。
- 涉及金额、折扣、退款的，写明"必须调用工具"，禁止模型自己算。
- 写清楚**必须转交顾问的情况**和**禁止事项**。
- 不确定的信息用保守措辞，并提示以官网或顾问核实为准。
