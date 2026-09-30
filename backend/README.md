# GoEuroOps 后端

「指北」留学咨询工作室的多 Agent 运营中枢后端。工作室专注瑞典、德国、荷兰、芬兰、丹麦的英语授课 CS 硕士申请，这个系统负责：

- 回答五国 CS 硕士的公开知识问题
- 介绍服务，并按公开规则计算报价
- 在用户同意后登记咨询线索
- 解释付款和退款政策，估算可退金额
- 需要个性化判断时，把用户交接给真人顾问

核心能力：

- **三路融合意图识别**：LLM few-shot + 本地向量 + 关键词，按细粒度业务意图路由
- **多 Agent 编排**：前台接待 / 留学咨询 / 费用售后 / 转顾问，支持复合问题并行协作
- **意图门控 RAG**：知识类问题和意图识别并行做投机预取，问候、投诉类不检索
- **Skills v2**：多信号路由（意图 + 关键词 + 语义样例 + 会话粘性）、渐进式披露、版本和命中统计、每个 Skill 自带回归用例
- **业务目录单一数据源**：`business/*.yaml` 驱动报价、退款计算、知识库种子和前端价目表
- Redis + ChromaDB 分层记忆、在线监控与路由降权、按场景校准的 LLM-as-Judge 评测

## 快速开始

推荐在仓库根目录一键启动全栈（见根目录 README）：

```bash
cp .env.example .env   # 填入 ANTHROPIC_API_KEY（兼容 Anthropic 协议的 DeepSeek 等也可以）
docker compose up -d --build
```

只跑后端测试：

```bash
pip install -r requirements.txt pytest
python -m pytest -q
```

## 主要接口

| 分类 | 接口 | 说明 |
|---|---|---|
| 对话 | `POST /chat`、`POST /chat/stream` | 一次性返回 / SSE 流式；响应带 `skills_applied`、`rag_gate`、工具调用 |
| 业务 | `GET /catalog` | 服务价目、优惠、付款退款政策、五国资料 |
| 线索 | `GET /leads`、`PATCH /leads/{id}` | 咨询线索和转顾问交接单；设置 `GOEUROOPS_ADMIN_TOKEN` 后需要 `X-Admin-Token` |
| Skills | `GET /skills`、`GET /skills/{id}`、`POST /skills/reload` | 查看、热加载 |
| | `POST /skills/match`、`GET /skills/evals` | 命中测试（得分明细）、跑命中回归用例 |
| 知识库 | `POST /search`、`POST /knowledge/add`、`POST /knowledge/upload`、`GET /knowledge/stats` | 检索（改写 → 召回 → 重排）、导入、统计 |
| 观测 | `GET /monitor`、`GET /trace/tool/{request_id}`、`GET /metrics` | 在线指标、单次请求的工具/Skill/门控明细、Prometheus |
| 评测 | `POST /eval/run` | 意图、Skill 路由、五维对话质量、工具调用检查；可选 RAG 门控对照实验 |

## 一次对话的链路

```text
/chat/stream
  ├─ MemoryManager 读取工作记忆 / 情景记忆 / 用户画像
  └─ AgentOrchestrator.run
       ├─ 并行：IntentRecognizer（LLM ∥ 向量 ∥ 关键词）
       │       RagGate.speculate（consulting / billing / general 三个 domain 的纯向量召回）
       ├─ 路由：主 Agent + 辅助 Agent（复合问题并行）
       ├─ RAG 门控：prefetch 注入知识片段 / on_demand 由模型自查 / off 移除检索工具
       ├─ Agent 执行：
       │    SkillManager.select → 注入命中的 Skills（附 references 目录）
       │    工具循环：报价 / 退款计算 / 线索登记 / 国家资料 / 知识检索 / read_skill_reference
       └─ 升级：转顾问意图或 CRITICAL → EscalationAgent 写交接单
  └─ 写回记忆；trace、Skill 命中、门控决策写入指标
```

## 目录

```text
api/main.py                    FastAPI 入口与全部接口
agents/agent_orchestrator.py   多 Agent 编排、路由、RAG 门控接入、Skill 注入
agents/tools.py                Agent 工具：处理函数 + 按角色组装的白名单；入参用 Pydantic 定义
tooling/schemas.py             工具入参模型（生成发给模型的 JSON Schema）和金额类工具的返回值模型（业务不变量）
tooling/gateway.py             工具网关：入参校验、超时、熔断、返回值校验、结果截断、指标；每个工具声明自己的执行策略
tooling/amount_guard.py        金额护栏：回复里的人民币金额必须有依据，否则这一句被替换（流式按句放行）
tooling/breaker.py             三态熔断器（网关和检索管理器共用）
mcp_server.py                  只读工具的 MCP 入口，给外部客户端用（依赖见 requirements-mcp.txt，不进后端镜像）
business/catalog.yaml          服务、价格、优惠、付款退款发票政策（改价格改这里）
business/countries.yaml        五国 CS 硕士申请资料（带官网链接和核实日期）
business/articles/*.md         知识库补充文章（时间线、材料清单、APS、FAQ 等）
business/pricing.py            报价与退款的确定性计算
business/lead_store.py         线索 / 交接单存储（Redis，故障时降级内存）
business/kb_seed.py            由业务目录生成知识库种子（带版本号，改动后自动重建）
core/intent_recognizer.py      三路融合意图识别与实体抽取
core/rag_gate.py               意图门控 RAG 与投机预取
core/skill_loader.py           Skills v2：加载、校验、多信号路由、渐进式披露、统计、回归
core/text_embedding.py         本地字符 n-gram 向量（意图识别与 Skill 语义匹配共用）
retrieval/knowledge_base.py    ChromaDB 知识库（bge-small-zh 向量 + 字面覆盖混合打分）
retrieval/embeddings.py        中文向量模型加载（fastembed / ONNX）
retrieval/manager.py            检索治理层：缓存、熔断、降级、查询改写、LLM 重排
memory/conversation_memory.py  Redis + ChromaDB 分层记忆
evaluation/evaluator.py        端到端评测
skills/                        业务规范（编写规范见 skills/README.md）
tests/                         单元测试（含所有 Skill 的命中回归用例）
```

## 工具层

所有 Agent 工具走同一个网关（`tooling/gateway.py`），每个工具在定义时声明自己的策略：

| 策略 | 含义 | 例子 |
|---|---|---|
| `timeout_s` | 超时；同步函数放进线程执行，超时才真的生效 | 报价 2 秒，检索 25 秒 |
| `side_effect` | `read` 只读 / `write` 写入 / `external` 访问外部服务 | 登记线索是 `write` |
| `breaker` | 是否熔断；只给依赖外部服务的工具开 | 登记线索（Redis）开，报价（纯计算）不开 |
| `max_result_chars` | 返回给模型的结果上限，超出时截断列表并标 `truncated` | 默认 12000 字符 |

执行顺序：白名单 → Pydantic 入参校验 → 熔断检查 → 超时内执行 → 返回值校验 → 截断 → 指标。
报价、退款的返回值还要满足业务不变量（应付不超过标价、可退不超过已付、退款必须标记人工核验），
不满足时不把这个结果交给模型，而是返回"请转顾问核算"。

**金额护栏**：模型回复里出现的人民币金额，必须能在本轮依据里找到（工具结果、用户原话、背景、知识库、Skill、价目表），
否则这一句话被替换成固定提示。流式输出时按句放行，不会先发出错误数字再撤回。
`GOEUROOPS_AMOUNT_GUARD=enforce`（默认）/ `warn`（只记录）/ `off`。命中记入请求的工具轨迹和 `goeuroops_amount_guard_total`。

**可观测**：`GET /tools/stats` 看每个工具的调用次数、各类结果、耗时和熔断状态；Prometheus 有
`goeuroops_tool_calls_total{tool,outcome}` 和 `goeuroops_tool_latency_seconds`。

**MCP 入口**：`mcp_server.py` 把只读的四个工具（价目、报价、国家资料、付款政策）用 MCP 对外提供，
和内部 Agent 共用同一份定义和网关。写入类工具不暴露。依赖单独装（`requirements-mcp.txt`）：mcp SDK 要求更新的
starlette，和后端固定的 fastapi 版本冲突，所以它在本地或单独环境运行，不进后端镜像。

```bash
pip install -r requirements-mcp.txt
claude mcp add goeuroops-studio -- python /绝对路径/backend/mcp_server.py
```

## 常用配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `GOEUROOPS_SKILL_MATCH_THRESHOLD` | 0.35 | Skill 命中阈值 |
| `GOEUROOPS_SKILLS_MAX_PROMPT_CHARS` | 6000 | 单次注入的 Skill 总字数上限 |
| `GOEUROOPS_RAG_GATE_ENABLED` | true | 关闭后所有意图退回"模型自行检索" |
| `GOEUROOPS_RAG_GATE_MIN_CONF` | 0.6 | 意图置信度低于该值时不预取也不关闭检索 |
| `GOEUROOPS_RAG_PREFETCH_TOP_K` | 3 | 预取片段数 |
| `GOEUROOPS_RAG_PREFETCH_MIN_SCORE` | 0.4 | 预取片段的混合分数下限 |
| `GOEUROOPS_EMBEDDING_MODEL` | BAAI/bge-small-zh-v1.5 | 知识库向量模型；`default` 使用 ChromaDB 默认英文模型 |
| `GOEUROOPS_ADMIN_TOKEN` | 空 | 设置后线索接口需要 `X-Admin-Token` |
| `GOEUROOPS_EVAL_DIM_THRESHOLD` | 0.75 | 评测建议的单维度阈值 |
| `GOEUROOPS_AMOUNT_GUARD` | enforce | 金额护栏：`enforce` 替换没有依据的金额 / `warn` 只记录 / `off` |

更详细的设计说明见 [wiki/Skills与意图门控RAG.md](wiki/Skills与意图门控RAG.md)。
