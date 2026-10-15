# GoEuroOps 架构设计

> 用途：做新需求前先读这份文档，弄清改动落在哪一层、会影响什么、要跑哪些检查。
> 范围：整个仓库（后端、前端、部署、评测）。模块内部的细节以代码和各模块的说明为准：
> [backend/README.md](../backend/README.md)、[Skills 与意图门控 RAG](../backend/wiki/Skills与意图门控RAG.md)、
> [Skill 编写规范](../backend/skills/README.md)、[部署](../deploy/DEPLOY.md)、[工作流程](WORKFLOW.md)。
>
> 最后核对：2026-10-13。代码改了结构时，请同步更新这份文档里对应的图和表。

## 目录

1. [系统定位](#1-系统定位)
2. [系统上下文](#2-系统上下文)
3. [部署架构](#3-部署架构)
4. [后端分层与依赖规则](#4-后端分层与依赖规则)
5. [一次对话的完整链路](#5-一次对话的完整链路)
6. [意图识别、路由与意图门控 RAG](#6-意图识别路由与意图门控-rag)
7. [Agent 角色与工具](#7-agent-角色与工具)
8. [记忆模块](#8-记忆模块)
9. [业务流程](#9-业务流程)
10. [业务目录：单一数据源](#10-业务目录单一数据源)
11. [Skills](#11-skills)
12. [知识库与检索](#12-知识库与检索)
13. [工具网关与金额护栏](#13-工具网关与金额护栏)
14. [数据存储清单](#14-数据存储清单)
15. [接口清单](#15-接口清单)
16. [前端](#16-前端)
17. [公开演示的安全与成本控制](#17-公开演示的安全与成本控制)
18. [观测与评测](#18-观测与评测)
19. [开发、测试与发布](#19-开发测试与发布)
20. [扩展指南：常见需求改哪里](#20-扩展指南常见需求改哪里)
21. [已知限制](#21-已知限制)

---

## 1. 系统定位

「指北 · Nordic CS Master Studio」是一家只做瑞典、德国、荷兰、芬兰、丹麦英语授课 CS 硕士申请的咨询工作室。
GoEuroOps 是它的多 Agent 运营中枢，AI 助手承担前台工作：

| 能做 | 不做（交给真人顾问） |
|---|---|
| 回答五国 CS 硕士的公开知识 | 个性化选校、录取概率判断 |
| 介绍服务，按公开规则**确定性计算报价** | 文书修改 |
| 用户同意后**登记咨询线索** | 实际发起退款、作废发票 |
| 解释付款退款政策，**估算可退金额** | 投诉处理、隐私删除的最终执行 |

设计上的三条底线，改代码时不能破坏：

1. **金额只能来自代码**：报价和退款由 `business/pricing.py` 计算，回复里没有依据的金额会被金额护栏替换。
2. **业务事实以系统记录为准**：价格、政策、五国资料来自 `business/*.yaml`，不靠模型记忆。
3. **任何依赖故障都要降级而不是报错**：Redis、ChromaDB、检索、记忆失败时，对话照常进行，只是少一点信息。

## 2. 系统上下文

```mermaid
flowchart LR
    student(["学生<br/>学生端 /"])
    owner(["站长 / 顾问<br/>工作室后台 /studio"])
    guest(["演示访客（面试官）<br/>/studio?guest=…，只读"])

    subgraph GoEuroOps
        fe["前端<br/>Vue 3 + Vite"]
        be["后端<br/>FastAPI + 多 Agent"]
    end

    llm[("LLM API<br/>Anthropic 协议<br/>线上为 DeepSeek")]
    emb[("向量 API<br/>线上为硅基流动 bge-large-zh<br/>本地可用 fastembed bge-small-zh")]
    mcp(["外部 MCP 客户端<br/>只读工具"])

    student --> fe
    owner --> fe
    guest --> fe
    fe -->|/api/*| be
    be --> llm
    be --> emb
    mcp -->|stdio| mcpserver["backend/mcp_server.py"]
```

- 模型走 Anthropic 协议，通过 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_MODEL` 切到 DeepSeek 等兼容服务。
- `mcp_server.py` 只暴露价目、报价、国家资料、付款政策四个只读工具，和内部 Agent 共用定义与网关。它的依赖和后端冲突，不进后端镜像。

## 3. 部署架构

线上是一台 1.6 GiB 内存的阿里云小服务器，和个人主页、Vparser 共用。

```mermaid
flowchart TB
    internet((公网)) -->|80/443| caddy

    subgraph server["阿里云服务器（docker compose：docker-compose.yml + deploy/docker-compose.prod.yml）"]
        caddy["Caddy<br/>HTTPS、basic auth、访客标记<br/>访问日志（去敏）"]
        frontend["frontend<br/>nginx 托管静态文件<br/>按 IP 限流"]
        backend["backend<br/>FastAPI（mem_limit 600m）<br/>ChromaDB 嵌入式"]
        redis[("redis<br/>AOF 持久化")]
        chroma_vol[("/app/data/chroma<br/>知识库 + 记忆向量")]
        caddy --> frontend
        frontend -->|/api 反向代理| backend
        backend --> redis
        backend --- chroma_vol
    end

    backend -->|HTTPS| llm[("LLM API")]
    backend -->|HTTPS| emb[("向量 API")]
```

| 环境 | 差别 |
|---|---|
| 本地 `docker compose up -d --build` | 起 backend、frontend、redis、chromadb（独立服务）、prometheus；向量默认本地 fastembed |
| 线上（prod 覆盖文件） | 加 Caddy；后端和前端不暴露端口；ChromaDB 改嵌入式（`CHROMA_HOST=""`）；向量走 API；chromadb、prometheus 放进 profile 默认不起；后端限内存 |

**服务器上不能构建镜像**（内存不够，会把 SSH 一起卡死）。发布方式见第 19 节。

## 4. 后端分层与依赖规则

```mermaid
flowchart TB
    subgraph L1["接口层 api/"]
        main["main.py<br/>启动装配（lifespan）、CLI"]
        routes["routes/<br/>chat · leads · skills · knowledge · evals · system · memory"]
        guard["demo_guard.py · quota.py<br/>访客只读、每日额度、花费上限"]
        state["state.py<br/>services 单例容器"]
    end

    subgraph L2["编排层 agents/"]
        orch["agent_orchestrator.py<br/>意图 → 路由 → 门控 → 执行 → 升级"]
        routing["routing.py<br/>规则路由，不调模型"]
        roles["roles.py<br/>4 个业务角色"]
        base["base.py<br/>工具循环、Skill 注入、金额护栏"]
        composer["composer.py<br/>多 Agent 合并"]
        tools["tools.py<br/>工具处理函数 + 角色白名单"]
    end

    subgraph L3["能力层"]
        core["core/<br/>意图识别 · RAG 门控 · Skills · 指标 · 用量 · 对话日志"]
        tooling["tooling/<br/>网关 · 入参/返回值模型 · 金额护栏 · 熔断"]
        retrieval["retrieval/<br/>知识库 · 向量模型 · 检索治理"]
        memory["memory/<br/>记忆（见第 8 节）"]
        monitor["monitor/<br/>在线监控、路由降权"]
        evaluation["evaluation/<br/>端到端评测、意图/记忆基准"]
    end

    subgraph L4["业务层 business/（单一数据源）"]
        catalog["catalog.yaml · countries.yaml · articles/"]
        pricing["pricing.py · catalog.py · lead_store.py<br/>kb_seed.py · domain_terms.py"]
    end

    L1 --> L2 --> L3 --> L4
    L1 --> L3
```

**依赖规则**

- 只能上层依赖下层。`business/` 不导入 `agents/`、`api/`；`core/` 不导入 `agents/`。`agents/routing.py` 读 `business/domain_terms.py` 是允许的，属于往下依赖。
- 运行时单例都放在 `api/state.py` 的 `services` 里，由 `api/main.py` 的 lifespan 装配。新组件也在这里装配，不要在模块导入时创建连接。
- 调用模型统一用 `core/llm_utils.make_client(…, source)`，用量会按来源记进 `core/llm_usage`，花费上限依赖这个统计。
- 拼进提示词的外部文本统一过 `safe_text`。

## 5. 一次对话的完整链路

```mermaid
sequenceDiagram
    autonumber
    participant U as 前端
    participant C as api/routes/chat.py
    participant M as MemoryManager
    participant O as AgentOrchestrator
    participant I as IntentRecognizer
    participant G as RagGate
    participant A as Agent（BaseAgent）
    participant T as 工具网关
    participant L as LLM

    U->>C: POST /chat/stream {user_id, conv_id, message}
    C->>C: 花费上限、每日额度检查（超了返回 429）
    C->>M: get_context(user, conv, query)
    M-->>C: 档案 + 以前的咨询记录 + 相关笔记 + 最近对话
    C->>O: run(Request)
    par 投机预取
        O->>G: speculate(message)：各 domain 纯向量召回
    and 意图识别
        O->>I: recognize(message, history)
        I->>L: few-shot 分类
        I-->>O: intent、urgency、置信度、实体
    end
    alt 低置信度 OTHER
        O-->>C: 澄清追问（不检索）
    else 正常
        O->>O: Router 选主 Agent + 协作 Agent
        O->>G: 门控决策：prefetch / on_demand / off
        O->>A: handle(req)（复合问题并行多个 Agent，再由 Composer 合并）
        A->>A: Skills 路由 → 拼 system prompt
        loop 最多 3 轮工具调用
            A->>L: messages + tools（流式）
            L-->>A: 文本 / tool_use
            A->>T: 执行工具（白名单、校验、超时、熔断）
            T-->>A: 工具结果
        end
        A-->>O: 回复（经金额护栏按句放行）
        O->>O: 升级检查 → EscalationAgent 写交接单
    end
    O-->>C: OrchestratorResult
    C->>M: 写入本轮问答；后台按时机整理记忆（第 8.2 节）
    C->>C: chat_log.record（站长后台按访客查看）
    C-->>U: SSE：token … done（带意图、Agent、工具、Skill、门控明细）
```

`/chat` 和 `/chat/stream` 共用 `_run_chat`。只有单 Agent 路径是逐 token 转发；澄清追问和多 Agent 合并是拿到全文后一次转发。专属 Agent 流到一半失败、改由 GeneralAgent 回答时，先发一个 `reset` 事件让前端清掉半段文字。

**提示词的组成**（`agents/base.py::_call_llm`），按顺序：

| 位置 | 内容 | 来源 |
|---|---|---|
| system | 角色提示词 + 命中的 Skills（含 references 目录） | `roles.py`、`core/skill_loader.py` |
| user / assistant 对 | `[背景信息]` 记忆上下文 | `memory/` |
| | `[结构化实体]` 意图识别抽出的实体 | `core/intent_recognizer.py` |
| | `[角色输入契约]` 本角色需要的输入和输出要求 | `roles.py` |
| | `[知识库上下文]` 门控预取的片段 | `core/rag_gate.py` |
| user | 本轮用户消息 | |

## 6. 意图识别、路由与意图门控 RAG

```mermaid
flowchart TD
    msg[用户消息] --> fuse

    subgraph fuse["三路融合意图识别（core/intent_recognizer.py）"]
        llm["LLM few-shot（权重 0.7）"]
        vec["本地字符 n-gram 向量（0.2）"]
        kw["关键词（0.1）"]
    end
    fuse --> res["intent / intent_group / urgency / 置信度 / 实体"]

    res --> q1{"低置信度且是 OTHER？"}
    q1 -->|是| clar[澄清追问，结束]
    q1 -->|否| q2{"urgency=CRITICAL 或<br/>转人工、找创始人、隐私？"}
    q2 -->|是| esc[EscalationAgent]
    q2 -->|否| score["意图给基础分 + 领域关键词、实体加分<br/>（agents/routing.py）"]
    score --> primary[最高分 = 主 Agent]
    score --> collab{"第二名 ≥ 0.45 且 ≥ 主 Agent 的 0.55？"}
    collab -->|是| par[并行协作，Composer 合并]
    collab -->|否| single[单 Agent]

    res --> gate{"意图门控 RAG（core/rag_gate.py）<br/>置信度 < 0.6 一律 on_demand"}
    gate -->|知识密集型意图| pre["prefetch：采用投机预取结果<br/>注入知识库上下文，检索工具保留"]
    gate -->|泛问题| od["on_demand：模型自己决定是否检索"]
    gate -->|问候、投诉、转人工、隐私| off["off：不检索，移除检索工具"]
```

- **意图体系**在 `core/intent_categories.py`，共 19 个值，包括通用大类（query、request、complaint 等）和细粒度业务意图（study_consult、service_inquiry、refund、data_privacy 等），细粒度优先。
- **意图到 Agent 的映射**在 `agents/routing.py` 的 `_GENERAL_INTENTS`、`_CONSULTING_INTENTS`、`_BILLING_INTENTS` 三个集合里。
- **领域关键词**集中在 `business/domain_terms.py`，意图识别和路由共用。
- **在线降权**：`monitor/` 每 10 秒把各 Agent **最近 5 分钟**（最多 50 次、至少 5 次）的成功率和首字耗时折算成惩罚写回编排器（`update_routing_penalties`）。同类有多个实例时选表现好的；每类只有一个实例，所以咨询或费用 Agent 的惩罚达到 0.5（`GOEUROOPS_MONITOR_FALLBACK_PENALTY`）时，Router 把这一类从本次路由里拿掉，问题交给 GeneralAgent，`routing_reason` 记 `monitor_fallback=[...]`。只靠延迟最多罚 0.4，单纯变慢不会触发；成功率掉到 65% 以下才会。被排除后没有新样本，窗口清空后惩罚归零，自动重新接流量。
- **执行失败回退**：专属 Agent 这一次失败时改由 GeneralAgent 回答，两次尝试的工具轨迹都保留，中间插一条 `agent_fallback`。如果失败前已经流出半段文字，SSE 先发 `reset`，前端清掉再接新回答。

## 7. Agent 角色与工具

| 角色 | 负责的意图 | 工具白名单（`roles.py` 的 `tool_scope`） | 必须转顾问的情况 |
|---|---|---|---|
| GeneralAgent 前台接待 | 问候、一般查询、服务进度、反馈、投诉、账号、其他 | `search_knowledge_base`、`inspect_request_context`、`suggest_required_fields`、`get_studio_profile` | 超出前台范围的个性化请求 |
| ConsultingAgent 留学咨询 | 五国知识、申请流程、服务与报价、预约 | `search_knowledge_base`、`lookup_country_admissions_overview`、`lookup_service_offering`、`quote_service_bundle`、`create_consultation_lead` | 个性化选校、文书修改、录取判断 |
| BillingAgent 费用售后 | 费用、退款、发票、付款异常 | `search_knowledge_base`、`check_payment_fields`、`calculate_refund`、`get_payment_policy`、`compare_amounts` | 实际发起退款；多付或付款成功但服务未确认；发票作废重开；对退款金额有异议 |
| EscalationAgent 转顾问 | 要求真人、找创始人、紧急、投诉升级、隐私删除 | `create_handoff_summary`（在 `handle()` 里直接调用，不经模型） | — |

- 命中的 Skill 带 references 时，本轮会动态加一个 `read_skill_reference` 工具，只能读本轮已命中 Skill 的资料。
- 门控为 `off` 时，`search_knowledge_base` 会从本轮工具列表里移除。
- 工具定义：入参是 `tooling/schemas.py` 里的 Pydantic 模型，发给模型的 JSON Schema 由模型自动生成；处理函数在 `agents/tools.py`。

## 8. 记忆模块

代码在 `backend/memory/`，入口是 `MemoryManager`（`manager.py`）。设计是客服系统常见的分层记忆，检索层用 A-Mem 笔记（arXiv 2502.12110）。
选型依据见 [CHANGELOG-2026-10 第 3 节](CHANGELOG-2026-10.md)：

- 纯分层（原话只按原文检索）在 LoCoMo 上比 A-Mem 差，所以检索层保留了 A-Mem 笔记；
- 客服能力（咨询记录、档案、证件号、删除）叠在上面。

### 8.1 分层

```mermaid
flowchart TB
    subgraph redis["Redis"]
        wm["① 会话状态 wm:{user}:{conv}<br/>每会话最多 60 条，提示词放最近 14 条<br/>24 小时过期"]
        prof["② 客户档案 mem:profile:{user}<br/>字段：值、出处原话、说话时间、来源、历史、待核实<br/>不过期"]
        cases["③ 咨询记录 mem:cases:{user}<br/>每个会话一份摘要：想解决什么、没解决的事、约好的下一步<br/>每人最近 10 份"]
        open["mem:open:{user}<br/>还没写咨询记录的会话"]
    end
    subgraph chroma["ChromaDB 集合 amem_notes_{向量模型}"]
        notes["④ 原话笔记（A-Mem）<br/>原话 + 关键词、标签、背景、事件时间<br/>链接、取代、背景的历史版本<br/>保留 180 天"]
    end
```

| 层 | 回答什么问题 | 谁来写 | 怎么取 |
|---|---|---|---|
| ① 会话状态 | 刚才在聊什么 | 每轮直接写，不调模型 | 当前会话最近 14 条 |
| ② 客户档案 | 用户是什么情况（专业、GPA、语言成绩、目标国家/城市、入学时间、预算、毕业时间、经历、偏好） | 模型提候选，**代码决定合并**；顾问可以改 | 整份放进提示词 |
| ③ 咨询记录 | 以前聊到哪、还有什么没办完、约了什么 | 会话停下来后由模型写 | 最近 2 份（不含当前会话） |
| ④ 原话笔记 | 档案以外的细节、原话出处、时间线 | 每句陈述立刻原样存；整理时补全 | 向量检索前 6 条，顺链接多取最多 3 条 |

**档案的合并规则**（`profile.py::apply_candidate`）：

- 后说的覆盖先说的，旧值进历史。
- 说话时间更早、但补全晚到的候选，只进历史。
- 顾问核实过的字段（`source=advisor`），模型不能覆盖；用户后来的不同说法记为“待顾问核实”。
- 经历和偏好可以有多条，全部保留。

### 8.2 写入和整理

```mermaid
sequenceDiagram
    autonumber
    participant C as chat.py
    participant M as MemoryManager
    participant R as Redis
    participant N as 笔记（ChromaDB）
    participant L as LLM

    C->>M: add_message(user, conv, 用户消息 / 助手回复)
    M->>M: 证件号换成占位符
    M->>N: 用户的陈述立刻存成原话笔记（同一句只刷新时间）
    M->>R: 写会话状态，会话记入 mem:open
    C->>M: schedule_after_turn（后台，不阻塞回复）
    Note over M: 触发整理：未整理笔记攒够 8 句<br/>或别的会话已停 2 分钟（换会话）<br/>或用户空闲 2 分钟（空闲计时器）
    M->>N: 每条新笔记取几条相似旧笔记
    M->>L: 一次调用：补全笔记 + 抽档案候选 + 给停下来的会话写咨询记录
    L-->>M: JSON
    M->>N: 写补全字段、双向链接、取代标记、旧笔记背景（只追加版本，重算向量）
    M->>R: 档案按规则合并；咨询记录增量更新；会话移出 mem:open
    M->>N: 删除这位用户 180 天前的笔记
```

- 当前会话还在进行时，只补全笔记，不写咨询记录。
- 同一会话停下来后又接着聊，下次整理只看新消息，在原摘要上更新。
- 模型输出无效时，原话保留，下次再整理；失败 2 次后放弃补全这几条，它们照样能检索。没写成的咨询记录留在 `mem:open`，下次再写。

### 8.3 读取：提示词里的记忆

`get_context` 并行读四层，`MemoryContext.to_prompt_text()` 按下面的顺序拼成 `[背景信息]`：

```text
[客户档案]（用户自己说的，未核实；付款、订单、报价以系统查询结果为准；括号里是以前的说法）
- 目标国家：瑞典（之前：荷兰）
[以前的咨询记录]（最近的在前）
- 2026-10-12｜想了解文书服务；还没解决：定预算；约好的下一步：周四 20:00 视频
[用户以前说过的相关原话]（按说话时间先后）
- [2025-07｜说于 2026-10-04] 我去年夏天考了雅思 6.0（背景：…）（后来有更新，以新的为准）
[最近对话]
user: …
assistant: …（助手回复截到 300 字）
```

总长超过 6000 字时，先去掉最早的笔记，再去掉较早的咨询记录；档案和最近对话不裁。

### 8.4 降级、隐私和顾问接口

- **降级**：Redis 读写超时 1 秒，失败就按“没有记忆”继续；笔记检索最多等 5 秒，向量查询报错时在这位用户的笔记里精确计算距离。任何失败都不影响回答。
- **证件号**：身份证号、护照号在写入任何一层之前就换成 `[证件号已隐去]`。
- **顾问接口**（需要后台口令或 `X-Admin-Token`，访客 403）：

| 接口 | 作用 |
|---|---|
| `GET /memory/{user_id}/profile` | 档案原样返回，带出处和历史 |
| `PUT /memory/{user_id}/profile/{slot}` | 顾问核实后改一个字段，之后模型不能覆盖 |
| `DELETE /memory/{user_id}` | 隐私删除：会话、档案、咨询记录、笔记、后台对话记录全部删除。线索和交接单是业务记录，由顾问在线索面板处理 |

- **交接单**：`build_handoff_summary` 把客户档案和以前的咨询记录写进 `background`，线索面板直接显示，顾问不用翻聊天记录。

### 8.5 评测

| 评测 | 测什么 | 命令 |
|---|---|---|
| 基础 + 难题 30 题 | 早期事实、滑出窗口、改主意、历史值、跨会话、不编造、多事实、长消息、长对话 | `python -m evaluation.memory_benchmark --runs 3` |
| 压力题 10 题 | 闲聊干扰、多会话、反复修改、相对时间、多跳 | `--set stress` |
| 客服题 8 题 | 回访接上没解决的事、助手约好的下一步、顾问承诺、证件号不被记住、不编造约定 | `--set service` |
| LoCoMo 70 题 | 公开长对话记忆数据集（英文） | `python -m evaluation.locomo_benchmark --data locomo10.json --convs 0,1 --per-cat 10` |

- 环境变量 `GOEUROOPS_MEMORY_MODULE` 可以切换评测的实现，方便和旧版同题对比：在旧版的工作树里设成 `memory.amem`。
- 当前结果（deepseek-v4-pro）：
  - 30 题、压力题、客服题都是 100%；
  - LoCoMo 3 次平均 49.0%，旧版 45.3%。
- 每次整理调用的 token 约为旧版的 2.5 倍。

**改记忆模块的规则**：新版必须和当前线上版本用同一个模型、同一套题对比，几项都不比旧版差才能上线。LoCoMo 单次波动有 10 个百分点左右，至少跑 3 次取平均。

## 9. 业务流程

### 9.1 报价

```mermaid
flowchart TD
    a[用户问价格或要报价] --> b[ConsultingAgent<br/>命中 service_sales Skill]
    b --> c{服务和数量说清楚了吗？}
    c -->|没有| d[追问：哪项服务、几个项目、签约时间、是否老学员介绍]
    c -->|清楚| e["quote_service_bundle<br/>business/pricing.py::quote_bundle"]
    e --> f["按 catalog.yaml 计算：标价 → 早鸟和团报取更优的一项<br/>→ 老带新可叠加 → 实付不低于标价的 85%"]
    f --> g{"返回值通过业务不变量校验？<br/>应付 ≤ 标价等"}
    g -->|否| h[不交给模型，返回“请转顾问核算”]
    g -->|是| i[模型按结果组织回复]
    i --> j[金额护栏：回复里的每个金额必须能在依据里找到]
```

### 9.2 线索登记

```mermaid
flowchart TD
    a[用户想预约或留联系方式] --> b{用户明确同意由顾问联系？}
    b -->|否| c[先征得同意，不登记]
    b -->|是| d["create_consultation_lead<br/>校验称呼、联系方式格式、阶段<br/>拒绝身份证号、护照号"]
    d -->|校验失败| e[说明哪里不对，请用户改]
    d -->|通过| f[("Redis goeuroops:lead:{id}<br/>Redis 不可用时降级到进程内存储")]
    f --> g[工作室后台「线索」面板<br/>状态 new → contacted → converted / closed]
```

### 9.3 退款估算

```mermaid
flowchart TD
    a[用户问能不能退、退多少] --> b[BillingAgent<br/>命中 billing_support Skill]
    b --> c["check_payment_fields：核验信息齐不齐<br/>（合同号、付款金额、付款时间、付款渠道）"]
    c -->|缺| d[追问缺的字段]
    c -->|齐| e["calculate_refund（business/pricing.py）<br/>按 catalog.yaml 的退款规则"]
    e --> f{"可退 ≤ 已付，且标记“需人工核验”？"}
    f -->|否| g[返回“请转顾问核算”]
    f -->|是| h[给出估算金额和依据，说明以顾问核实为准]
    h --> i{"用户要实际发起退款，或对金额有异议？"}
    i -->|是| j[转顾问，见 9.4]
```

### 9.4 转顾问（升级）

```mermaid
flowchart TD
    a["触发：转人工/找创始人/隐私意图，<br/>urgency=CRITICAL，或 Agent 回复判定需要升级"] --> b[EscalationAgent]
    b --> c["build_handoff_summary<br/>request_id、user/conv、原因、意图、紧急度、实体、最后一句话<br/>background：客户档案 + 以前的咨询记录"]
    c --> d{"同一会话已有未关闭的交接单？"}
    d -->|是| e[追加到原交接单的 followups（最多 10 条）]
    d -->|否| f[("新建交接单 goeuroops:lead:H…")]
    e --> g[后台「线索」面板，顾问跟进]
    f --> g
    b --> h["固定回复（不调模型）：交接单编号、转交原因、响应时效、时差提醒；<br/>隐私请求说明由顾问核实身份后处理；提醒不要发证件号、银行卡号和密码"]
```

## 10. 业务目录：单一数据源

`business/catalog.yaml`（服务、价格、优惠、付款退款发票政策）和 `business/countries.yaml`（五国资料，带官网链接和核实日期）驱动：

```mermaid
flowchart LR
    yaml["catalog.yaml<br/>countries.yaml<br/>articles/*.md"] --> catalog_py["catalog.py<br/>Pydantic 校验<br/>启动时校验一次，写错直接启动失败"]
    catalog_py --> pricing["pricing.py<br/>报价、退款"]
    catalog_py --> seed["kb_seed.py<br/>知识库种子（带内容版本，变了自动重建）"]
    catalog_py --> api["GET /catalog<br/>前端价目表、学生端"]
    catalog_py --> guard["金额护栏的依据之一"]
    catalog_py --> routing["routing.py<br/>服务名用于去除歧义"]
```

改价格、政策、国家资料只改 YAML。线上 `business/` 挂载进容器，改完 `docker compose restart backend` 即可，不用重建镜像。

## 11. Skills

Skill 是可热加载的业务规范（话术、流程、边界、升级条件），结构对齐 Anthropic Agent Skills 的渐进式披露：

```text
backend/skills/<skill_id>/
├── SKILL.md          front matter（agents、intents、keywords、examples、version）+ 核心规则，命中后注入 system prompt
├── references/*.md   详细资料，只列目录，模型需要时用 read_skill_reference 读
└── evals/cases.json  命中回归用例（该命中 / 不该命中的句子）
```

| Skill | 绑定 Agent | 绑定意图 |
|---|---|---|
| `studio_brand_voice` | 全部（`mode: always`） | — |
| `general_reception` | general | greeting、query、request、feedback、complaint、service_progress、account、other |
| `study_consulting` | consulting | study_consult、application_process |
| `service_sales` | consulting | service_inquiry、booking |
| `billing_support` | billing | billing、refund、invoice、payment_issue |

**路由是确定性的**，不额外花一轮模型调用：意图绑定（0.5）+ 关键词 + 语义样例（本地 n-gram 向量）+ 会话粘性，超过阈值（默认 0.35）就注入，总字数上限 6000。热加载用 `POST /skills/reload`。

## 12. 知识库与检索

```mermaid
flowchart LR
    q[查询] --> rm["RetrievalManager（retrieval/manager.py）<br/>TTL 缓存 → 熔断 → 查询改写（LLM）"]
    rm --> kb["KnowledgeBase（retrieval/knowledge_base.py）<br/>ChromaDB 集合 kb_{向量模型}"]
    kb --> score["混合打分：向量相似度 + 字面覆盖"]
    score --> rr["LLM 重排（可选）"]
    rr --> out[片段 + 来源]
    rm -.故障.-> fb[降级结果：“知识库暂时不可用”]
```

- **两条入口**：门控 `prefetch` 时在意图识别的同时做纯向量召回（不改写、不重排），用来压延迟；模型主动调用 `search_knowledge_base` 时走完整链路。
- **降级结果不当证据**：知识库超时、出错或熔断时，两条入口都把降级条目滤掉。预取得到空结果；检索工具返回 `success=false` 和“知识库暂时不可用”，由模型告诉用户稍后再问或请顾问确认。
- **向量模型**（`retrieval/embeddings.py`）：本地默认 fastembed bge-small-zh（ONNX，约 90MB）；线上用向量 API（bge-large-zh），省内存。换模型时集合名跟着变，两个模型的数据不会混在一起。
- **种子**由 `business/kb_seed.py` 从业务目录生成，带内容版本号，内容变了启动时自动重建；后台也能上传文档（`/knowledge/add`、`/knowledge/upload`）。

## 13. 工具网关与金额护栏

所有 Agent 工具都走 `tooling/gateway.py`，执行顺序固定：

```mermaid
flowchart LR
    a[模型发起 tool_use] --> b{在本角色白名单里？}
    b -->|否| x1[拒绝]
    b -->|是| c[Pydantic 入参校验]
    c -->|失败| x2[把校验错误返回给模型]
    c --> d{熔断器放行？}
    d -->|否| x3["返回“暂时不可用，请稍后或转顾问”"]
    d -->|是| e["在超时内执行<br/>同步函数放线程"]
    e --> f["返回值校验<br/>报价 / 退款检查业务不变量"]
    f -->|不满足| x4[返回“请转顾问核算”]
    f --> g["超过 max_result_chars 时截断<br/>标 truncated"]
    g --> h[记指标、写 trace]
```

每个工具在定义时声明 `timeout_s`、`side_effect`（read / write / external）、`breaker`、`max_result_chars`。

**熔断器**（`tooling/breaker.py`，网关和检索管理器共用）：连续失败 5 次打开，60 秒后半开，**只放行一个探测请求**，探测成功关闭、失败重新打开；探测被取消、迟迟不回时，过 60 秒再放一个。每次状态切换都换一代 ticket，调用结束时交回 ticket，旧 ticket 的结果直接忽略，所以熔断前发出、熔断后才返回的慢请求不会把熔断器提前关上。

**金额护栏**（`tooling/amount_guard.py`）：回复里出现的人民币金额，必须能在本轮依据里找到。依据包括工具结果、用户原话、背景、知识库片段、Skill 和价目表。找不到时，这一句被替换成固定提示。流式输出按句放行，不会先发出错误数字再撤回。由 `GOEUROOPS_AMOUNT_GUARD` 控制：`enforce`（默认）/ `warn` / `off`。

## 14. 数据存储清单

| 存储 | 键 / 集合 | 内容 | 保留 | 写入方 |
|---|---|---|---|---|
| Redis | `goeuroops:lead:<id>`、`goeuroops:leads` | 线索和交接单、索引 | 不过期 | `business/lead_store.py` |
| Redis | `goeuroops:chatlog` | 每轮问答（站长后台按访客查看） | 最近 2000 条 | `core/chat_log.py` |
| Redis | `goeuroops:usage:<北京日期>` | 按天、按来源的调用数、token、费用估算 | 120 天 | `core/llm_usage.py` |
| Redis | `goeuroops:quota:chat:<日期>`、`goeuroops:quota:guest_eval:<日期>` | 每日对话、访客评测次数 | 当天 | `api/quota.py` |
| Redis | `wm:<user>:<conv>` | 会话消息（最多 60 条） | 24 小时 | `memory/` |
| Redis | `mem:profile:<user>`、`mem:cases:<user>`、`mem:open:<user>` | 客户档案、咨询记录、待写记录的会话 | 档案不过期；记录每人 10 份 | `memory/` |
| ChromaDB | `kb_<向量模型>` | 知识库片段 | 种子按版本重建 | `retrieval/knowledge_base.py` |
| ChromaDB | `amem_notes_<向量模型>` | 原话笔记（A-Mem） | 180 天 | `memory/` |
| 文件 | `business/*.yaml`、`business/articles/`、`backend/skills/` | 业务目录、文章、Skills | Git | 人工 |
| 文件 | `/app/data/eval/baseline.json` | 评测基线 | 卷 | `evaluation/evaluator.py` |
| 文件 | Caddy 访问日志 `/data/logs/` | 去掉口令和 Cookie 的访问记录 | 90 天 | Caddy |

- 线上 Redis 开了 AOF，数据在 `redis-data` 卷里。
- ChromaDB 线上是嵌入式，数据在后端的数据卷里。
- 生产数据只通过应用接口查看，不直接读库。

## 15. 接口清单

网关路径前缀是 `/api`，下表省略。设置 `GOEUROOPS_ADMIN_TOKEN` 后，管理类接口需要请求头 `X-Admin-Token`；线上还有 Caddy basic auth（第 17 节）。

| 分类 | 接口 | 说明 | 访客可写 |
|---|---|---|---|
| 对话 | `POST /chat`、`POST /chat/stream` | 一次性 / SSE 流式；响应带意图、Agent、工具、Skill、门控明细 | ✓（计入额度） |
| 业务 | `GET /catalog` | 价目、优惠、政策、五国资料 | — |
| 线索 | `GET /leads`、`PATCH /leads/{id}` | 线索和交接单；访客看到的联系方式打码 | ✗ |
| 记忆 | `GET /memory/{user_id}/profile`、`PUT /memory/{user_id}/profile/{slot}`、`DELETE /memory/{user_id}` | 看档案、顾问核实改字段、隐私删除 | ✗ |
| Skills | `GET /skills`、`GET /skills/{id}`、`POST /skills/reload`、`POST /skills/match`、`GET /skills/evals` | 查看、热加载、命中测试、跑回归用例 | 仅 match |
| 知识库 | `POST /search`、`POST /knowledge/add`、`POST /knowledge/upload`、`GET /knowledge/stats` | 检索、导入、统计 | 仅 search |
| 评测 | `POST /eval/run` | 意图、Skill 路由、五维质量、工具调用、门控对照 | ✓（每天限次，只跑内置用例） |
| 观测 | `GET /monitor`、`GET /trace/tool/{request_id}`、`GET /trace/tools`、`GET /tools/stats`、`GET /metrics` | 在线指标、单次请求明细、Prometheus | — |
| 系统 | `GET /session`、`GET /health` | 当前身份（guest / owner）、健康检查 | — |

**新增写接口默认只对站长开放**：访客能调用的写接口是白名单（`api/demo_guard.py::GUEST_WRITE_ALLOWED`），不在名单里的一律 403。

## 16. 前端

```text
frontend/src/
├── Root.vue                路径以 /studio 结尾进后台，否则进学生端；两边各自一个 chunk，全局样式互不影响
├── student/                学生端：StudentSite（五国和服务介绍）、ChatPanel（对话）、北欧天空动画
├── App.vue                 工作室后台：对话调试 / 线索 / 服务价目 / Skills / 知识库 / 评测 六个标签
├── components/             LeadsView、CatalogView、SkillsView、LangSwitch
└── lib/                    backends.js（API 调用、设置存 localStorage）、i18n.js + en.js（中英双语）、labels.js、markdown.js
```

- 后台启动时调 `GET /session`，访客模式下禁用写操作，并显示每日额度提示。
- 本地开发：`cd frontend && npm run dev`，Vite 把 `/api` 代理到 `localhost:8000`。
- 文案改动要同时改中文和 `lib/en.js`。CI 会构建前端，导入写错会直接失败。

## 17. 公开演示的安全与成本控制

```mermaid
flowchart TD
    req[公网请求] --> enter{"Caddy：入口路径"}
    enter -->|"/studio?guest=口令"| setc["写 30 天访客 Cookie，跳回 /studio"]
    enter -->|"/studio/owner"| own["basic auth，清掉访客 Cookie"]
    enter -->|其他| mark{"带有效访客 Cookie<br/>且没有 Authorization 头？"}
    mark -->|是| guest["请求加 X-Studio-Role: guest，免密码"]
    mark -->|否| pub{"公开路径？<br/>/、静态资源、/api/chat*、/api/catalog"}
    pub -->|是| open[直接放行]
    pub -->|否| auth["basic auth（站长）"]
    guest --> be["后端 demo_guard：写操作白名单，联系方式打码"]
    open --> rl["nginx：每 IP 每分钟 6 次对话"]
    rl --> quota["后端：每日对话上限（默认 200）<br/>每日费用上限（默认 3 元，按 llm_usage 估算）<br/>超了返回 429，北京时间零点恢复"]
```

- 外部请求自带的 `X-Studio-Role` 会先被 Caddy 删掉，所以访客身份无法伪造。
- 访客链接的口令少于 16 位时，整个访客通道失效。
- 换口令即可让旧的访客链接作废。

## 18. 观测与评测

**在线观测**

- 每次回答都带 `skills_applied`、`rag_gate`、`tools_used`、`routing_reason`。完整明细可以用 `GET /trace/tool/{request_id}` 查到。
- `monitor/performance_monitor.py` 定期采集 Agent 和工具的统计，用 Z-score 检测突变，可选 Webhook 告警，并把最近 5 分钟的成功率、首字耗时写回路由（第 6 节）。

几个字段容易读错：

| 字段 | 实际含义 |
|---|---|
| `knowledge_used` | 本轮门控预取到了片段，或检索工具真的查到了内容（查空、知识库降级都不算）；不代表回答真的用了这些片段 |
| `primary_agent` 和 `agent_type` | 前者是路由的决定，后者是实际给出回答的 Agent；专属 Agent 失败回退后两者不同 |
| trace 里的 `success` 和 `result_success` | 前者表示网关执行没出错（白名单、校验、超时、熔断都过了），后者是工具自己返回的 `success`；知识库查空或降级时前者为真、后者为假 |
| trace 里的 `amount_guard`、`agent_fallback` | 不是真的工具调用，是金额护栏改写、专属 Agent 回退的事件记录 |
- 主要的 Prometheus 指标有 `goeuroops_tool_calls_total{tool,outcome}`、`goeuroops_tool_latency_seconds`、`goeuroops_amount_guard_total`。

**离线评测**（`backend/evaluation/`）

| 工具 | 测什么 | 怎么跑 |
|---|---|---|
| `evaluator.py` | 意图准确率、Skill 路由准确率、五维 LLM-as-Judge（按场景的期望行为校准，含边界合规）、工具调用检查、门控 A/B、与基线对比 | 后台「评测」或 `POST /eval/run` |
| `intent_benchmark.py` | 意图识别基准 | `python -m evaluation.intent_benchmark` |
| `memory_benchmark.py`、`locomo_benchmark.py` | 记忆：自建 48 题 + LoCoMo（见 8.5） | 见 8.5 |
| `skills/*/evals/cases.json` | 每个 Skill 的命中回归 | `GET /skills/evals`，单元测试里也会跑 |

评测报告写在 `backend/evaluation/reports/`。报告里会记下所用模型，**不同模型的数字不能直接比较**。

`/eval/run` 的口径：
- **评测用单独的编排器**：线上的 Agent 统计、监控降权和 trace 不被评测流量污染；报价登记、转顾问生成的线索和交接单只写进内存。知识库和检索管理器（缓存、熔断）仍然共用，也会消耗真实的模型额度。
- **Judge 评分失败不计分**：输出为空、被截断、JSON 不合法、字段缺失或不在 0–1 都算失败，最多试两次（401/403 不重试）。失败的样本不进均分和通过率，`pass_rate` 的分母是 `scored`。
- **报告状态**：`complete` / `partial`（部分对话评分失败）/ `failed`（对话评分全部失败）/ `empty`。只有 `complete` 才做回归对比、才更新基线。
- **回归只在同一套件之间比**：`suite_id` 是用例、Judge 模型和评测选项的指纹；换了用例或模型就不比，旧基线没有 `suite_id`，第一次跑完整评测后才开始对比。

## 19. 开发、测试与发布

```mermaid
flowchart LR
    a[需求] --> b["分支开发<br/>本地测试 + 评测"]
    b --> c[PR]
    c --> d{"CI（.github/workflows/ci.yml）<br/>ruff + pytest · MCP 测试<br/>前端构建 · 部署文件语法"}
    d -->|通过| e[squash 合并到 main]
    e --> f["deploy/deploy.sh<br/>或 GitHub Actions「Deploy」手动触发"]
    f --> g["线上验证：真实请求"]
    f -.失败.-> r[自动回滚到 :prev]
```

- **main 受保护**：所有改动走 PR → CI → squash。
- **测试**：`cd backend && python -m pytest -q`，不需要真的 Redis 或模型，测试里用的是替身。另外跑 `ruff check .`。
- **发布**：`deploy/deploy.sh [backend|frontend|rollback]`。
  - 先检查本地改动已全部提交、推送，再让服务器 `git pull` 到同一个提交。
  - 前端在本机构建，把 dist 传到服务器。
  - 后端在固定的 `:base` 镜像上叠一层代码，得到新的 `:latest`；旧的 `:latest` 记为 `:prev`。
  - 只重建改动的容器，等健康检查通过，再从公网访问一次。
  - 任何一步失败都自动回滚。
- **依赖变了**（`requirements.txt`、Dockerfile、nginx 配置）不能用叠一层的办法，脚本会停下来提示，需要换 `:base` 镜像。
- **改了什么、怎么生效**：

| 改动 | 生效方式 |
|---|---|
| `backend/skills/` | 已挂载，`POST /skills/reload` |
| `backend/business/` | 已挂载，重启后端；知识库种子按版本自动重建 |
| 其他后端或前端代码 | 走发布流程 |
| `.env` | 重建对应容器 |

## 20. 扩展指南：常见需求改哪里

| 需求 | 要改的地方 | 必须一起做的事 |
|---|---|---|
| 改价格、优惠、退款规则 | `business/catalog.yaml` | 跑 `tests/test_business.py`；规则变了要补用例 |
| 加一项服务 | `catalog.yaml` | 服务名会自动进入路由的去歧义词表；检查 `service_sales` Skill 的话术 |
| 更新五国资料 | `business/countries.yaml`（带官网链接和 `last_verified`） | 知识库种子自动重建 |
| 加一个意图 | `core/intent_categories.py`、`business/domain_terms.py` 关键词、意图识别的 few-shot 示例、`agents/routing.py` 的映射集合、`core/rag_gate.py` 的门控策略 | 跑意图基准；相关 Skill 的 `intents` 也要加 |
| 加一个工具 | `tooling/schemas.py` 入参（和金额类的返回值）模型 → `agents/tools.py` 处理函数和 `AgentToolSpec`（声明超时、副作用、熔断）→ `roles.py` 的 `tool_scope` | 写入类工具要考虑幂等和降级；涉及金额的一定要有返回值不变量；单元测试 |
| 加一个 Agent 角色 | `agents/base.py::AgentType`、`roles.py` 新类（提示词、契约、工具、转交条件）、`agent_orchestrator.py` 注册、`routing.py` 映射、`core/skill_loader.py::VALID_AGENTS` | 评测用例覆盖新角色 |
| 加或改一个 Skill | `backend/skills/<id>/`（按 `skills/README.md`） | 必须带 `evals/cases.json`；`GET /skills/evals` 全过 |
| 加一个后台写接口 | `api/routes/<模块>.py`，在 `api/main.py` 注册；需要鉴权时用 `leads._require_admin` | 访客默认 403，确实要开放才加进 `GUEST_WRITE_ALLOWED`；涉及个人信息的给访客打码 |
| 加一个新的模型调用 | 用 `make_client(…, source)` | 新来源会出现在用量统计里，花费上限自动覆盖 |
| 改前端文案 | 对应 `.vue` + `lib/en.js` | CI 构建通过 |
| 改部署 | `deploy/`（Caddyfile、prod compose、deploy.sh） | CI 会检查语法；先在本地 compose 验证 |
| 改记忆 | `backend/memory/` | 按 8.5 的规则和线上版本同题对比；`tests/test_memory.py` |

## 21. 已知限制

完整的差距评估和计划见 [ROADMAP-commercial.md](ROADMAP-commercial.md)。和架构相关的主要几条：

- **单机、单进程**：编排器的统计、trace、记忆的后台整理和空闲计时器都在进程内，不能直接横向扩成多副本。要扩容时，这些状态得先移到 Redis 或独立的任务队列。
- **鉴权是演示级别**：站长靠 Caddy basic auth 和可选的 `X-Admin-Token`，没有顾问账号体系，CORS 是 `*`。
- **价目和五国资料是草案**，需要创始人逐条核实。
- **评测只隔离了编排器**：线索写进内存，但知识库、检索缓存和熔断器跟线上共用，评测期间检索出问题会影响线上；评测仍是同步接口，用例多了要等。
- **意图基准是用本地 bge-small-zh 测的**，线上换成向量 API 之后需要重测。
