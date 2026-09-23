# 2026.09 改动说明

本文记录项目从通用智能客服系统（原名 EchoMind）演进为 GoEuroOps 的过程。文中提到的 `EchoMind/`、`EchoMindFrontend/` 分别对应现在仓库里的 `backend/`、`frontend/`，`ECHOMIND_*` 环境变量对应现在的 `GOEUROOPS_*`。

> 注意：第 1–8 节按改名前的原始名称和路径书写；第 9 节说明改名与整理过程。

---

## 1. 让项目跑起来 + 去掉前端的 Java 部分

**背景**：Java 版本目录已被删除，前端却还保留着 Java / Python 双后端切换。

| 改动 | 文件 |
|---|---|
| `.env` 填入 DeepSeek 的 `ANTHROPIC_API_KEY`（该文件在 `.gitignore` 中，不会入库） | `EchoMind/.env` |
| 修复 `echomind` 服务健康检查：镜像里没有 `curl`，导致容器一直 unhealthy、nginx 起不来。改成和 Dockerfile 一致的 Python `urllib` 写法 | `EchoMind/docker-compose.yml` |
| 去掉后端切换器，`backends.js` 简化为单一后端，接口函数不再传 `type` 参数 | `EchoMindFrontend/src/lib/backends.js`、`src/App.vue` |
| 去掉健康检查失败时"回退到 Java"的逻辑 | `src/App.vue` |
| 代理路径 `/api/python` + `/api/java` 合并成 `/api` | `vite.config.js`、`public/runtime-config.js`、`docker/entrypoint.sh`、`docker/nginx-gateway.conf` |
| 删除 `echomind-java` 服务及其依赖 | `EchoMindFrontend/docker-compose.yml` |
| README 去掉 Java 相关说明 | `EchoMindFrontend/README.md` |

---

## 2. 修复评测分数为 0 的根因（DeepSeek 默认开启思考模式）

**现象**：`/eval/run` 的 `pass_rate` 为 0，四个质量维度全部恰好等于 0.5。

**根因**：DeepSeek v4 系列在 Anthropic 兼容端点上默认开启扩展思考，先输出 `thinking` 块再输出 `text` 块。项目里大量调用的 `max_tokens=256`，被思考过程耗尽，拿不到 `text` 块，`extract_text_content` 返回空串，`json.loads('')` 报错，LLM Judge 就回落到默认值 0.5（低于 0.75 及格线）。换成 `deepseek-v4-flash` 同样复现，所以不是模型选择问题。

**修复**：
- `core/llm_utils.py` 新增 `NO_THINKING_KWARGS = {"extra_body": {"thinking": {"type": "disabled"}}}`。
- 应用到全部 10 处 LLM 调用：`core/intent_recognizer.py`（1）、`mcp/tool_manager.py`（查询改写、重排，2）、`evaluation/evaluator.py`（Judge，1）、`memory/conversation_memory.py`（3）、`agents/agent_orchestrator.py`（Agent 主循环 + Composer，2）。

**效果**：`pass_rate` 0.0 → 1.0；意图准确率 72.7% → 90.9%；所有 `judge_failed` 变为 `False`。同时去掉了思考 token，响应也更快。

---

## 3. 流式输出（SSE）

**背景**：原来所有 LLM 调用都是 `messages.create()`，整条链路跑完才一次性返回，前端也没有流式处理能力。

**后端**
- `agents/agent_orchestrator.py`：新增 `OnDelta` 回调类型；`BaseAgent._complete()` 有回调时走 `client.messages.stream()`，逐 token 转发文本；`handle` / `_call_llm` / `_execute` / `run` / `run_parallel` / `EscalationAgent.handle` 都增加可选的 `on_delta` 参数。单 Agent 路径是真正逐 token；澄清追问和多 Agent 并行汇总两条边缘路径拿到完整文本后一次性转发。
- `api/main.py`：新增 `POST /chat/stream`（`text/event-stream`），事件类型 `token` / `done` / `error`，`done` 事件携带完整 `ChatResponse` 字段。
- `config/nginx/nginx.conf`：新增 `location /chat/stream`，关闭缓冲、读超时放宽到 300s。

**前端**
- `src/lib/backends.js`：新增 `streamChat()`（fetch + ReadableStream 手写 SSE 解析），删除不再使用的 `requestChat`。
- `src/App.vue`：`sendMessage()` 改为先插入助手占位消息，边收 token 边追加，`done` 后再补全路由、意图、trace、耗时。

---

## 4. Markdown 渲染

**背景**：助手回复被当成纯文本渲染（`<p>{{ content }}</p>`），`###`、`**` 等语法会原样显示。

- 新增依赖 `markdown-it`，新增 `src/lib/markdown.js`（`html: false`，避免 XSS）。
- `App.vue` 改用 `v-html="renderMarkdown(...)"`，`styles.css` 增加标题、列表、加粗、代码块、表格、引用等样式。

---

## 5. 业务改造：通用客服 → 留学咨询运营助手

**定位**：只做前台答疑和线索承接（介绍工作室服务、回答瑞典/德国/荷兰/芬兰/丹麦英语授课 CS 硕士的公开知识、个性化需求引导预约人工顾问），**不代做选校决策，不代写文书**。只改造 Technical Agent 这一个槽位，General / Billing / Escalation 及其 Skill、知识库内容保持不变。

| 改动 | 说明 |
|---|---|
| `AgentType.TECHNICAL` → `CONSULTING`；`TechnicalAgent` → `ConsultingAgent` | 全仓库同步改名，`ECHOMIND_TECHNICAL_MODEL` 等派生名称随之变化 |
| 新意图 `STUDY_CONSULT` / `APPLICATION_PROCESS` / `SERVICE_INQUIRY` | 替换 `TECHNICAL` / `TECHNICAL_LOGIN` / `TECHNICAL_CRASH`，保持"1 个大类 + 2 个细分类"结构；同步更新 few-shot 模板、关键词兜底、LLM 分类提示句 |
| 实体抽取 | `error_code` 换成 `country`（瑞典/德国/荷兰/芬兰/丹麦，中英文） |
| 新工具 | `lookup_country_admissions_overview`（五国参考信息）、`lookup_service_offering`（选校咨询 / 文书辅导服务介绍）。均为静态数据并带免责声明，不冒充实时信息 |
| `ConsultingAgent` 画像与 system prompt | 明确"能做什么 / 不能做什么"，`handoff_conditions` 列出必须转人工的场景 |
| 路由打分 | `technical_kws` → `consulting_kws`，`error_code` 加分 → `country` 加分，`_collaboration_targets` 同步 |
| Skills | 删除 `skills/technical_support/`，新增 `skills/study_consulting/SKILL.md`（`agents: consulting`），更新 `skills/README.md` |
| 知识库种子 | 移除"技术故障排查"，新增 7 篇留学占位文档（工作室服务介绍、五国各一篇概览、申请材料与时间线），均带"示例参考信息"声明。其余 5 篇电商文档不动 |
| 评测默认用例 | 技术相关的 2 条意图用例、1 条对话用例换成留学场景 |
| 测试 | `tests/test_agent_orchestrator.py` 全部改名并重新指向新工具/新实体 |
| 前端快捷提问 | "技术排查"按钮换成"留学咨询" |

**验证**：`pytest` 通过；手测三个场景符合预期——公开知识问题带免责声明并主动引导预约；问收费不编造价格；要求"帮我选校"时明确拒绝给结论并转人工。退款 / 问候链路不受影响。

> 知识库里的留学内容**只是占位示例**，需要替换成真实核实过的数据。

---

## 6. 前端视觉重设计

**品牌**：页面对外品牌为「指北 · True North」（"指北"是"指南"的双关，也贴合"指向北欧"）。注意这是产品界面里的品牌，和项目名（见第 9 节）是两回事。

**视觉**
- 从深色 SaaS 仪表盘风格改为浅色北欧风，再进一步调整为瑞典 / 宜家色系：蓝 `#0058A3` 作主色，黄 `#FFDB00` 只在一个地方饱和使用（空状态星标），信息类高亮用压暗的金色，用户消息气泡为浅黄，背景 `#F6F6F2`。
- 字体：标题 Space Grotesk，正文 Manrope，数据 / trace / JSON 用 IBM Plex Mono。
- 去掉所有 ALL-CAPS 英文小标签（`CONVERSATION LAB`、`SESSION` 等），卡片由渐变+阴影改为扁平白底+细边框。
- 清理死代码：`.kicker`、Java 切换遗留的 `.backend-tabs`。
- 页面标题、H1、介绍文案改为直接说明用途；知识库页 / 评测页文案同步收紧。

**动效（克制使用，只回应动作或强调一处）**：流式打字光标、发送后等待首 token 时的三点跳动、新消息滑入、响应耗时数字滚动、评测分数条填充过渡、空状态星标缓慢呼吸。没有给每个卡片加淡入或悬浮效果。

---

## 7. RAG 检索链路修复

对 RAG 做了一次排查，发现并修复以下问题（均用真实请求复现过）：

| 问题 | 修复 |
|---|---|
| **去重失效**：`search_with_rewrite` 用整条结果（含随查询变化的 `score`）做哈希，同一段内容被多个子查询命中会被当成不同结果。实测 `top_k=3` 返回同一篇文档 3 遍 | 改按"标题 + 内容"去重，重复命中保留分数高的一条 |
| **无相关性过滤**：完全无关的问题（如"今天天气怎么样"）也会返回 5 条结果，score 甚至为负 | `_rerank` 由"只排序"改为逐条打 0–10 相关分，低于 `RERANK_MIN_RELEVANCE = 4.0` 的丢弃；过滤到空时返回 `success: false`，诚实地告诉 Agent 没有相关内容 |
| **知识库不分业务域**：所有文档在同一个集合里，没有任何隔离 | 文档增加 `domain` 元数据（`consulting` / `billing` / `general` / 缺省 `shared`）；检索时 `where={"domain": {"$in": [domain, "shared"]}}`；`AgentOrchestrator.set_shared_tools` 现在为每种 Agent 绑定各自 domain 的 RAG 工具；`/knowledge/add`、`/knowledge/upload`、`/search` 增加可选 `domain` 参数 |
| **死代码**：`api/main.py` 里的 `_build_knowledge_context` / `_should_use_knowledge`（约 60 行）从未被调用 | 已删除 |

**验证**：瑞典查询由"同一篇 ×3"变为 2 篇不同的相关文档；无关查询返回空；退款问题限定在 `consulting` 域返回空、在 `billing` 域能命中；新增 1 个测试，`pytest` 共 11 项通过；评测 `pass_rate` 87.5%、意图准确率 90.9%。

**评估后没有做的**
- **更换中文 embedding 模型**（当前是 ChromaDB 默认的 `all-MiniLM-L6-v2`，实测"瑞典"查询原始向量排第一的是荷兰）：需要引入 `sentence-transformers` / torch，镜像体积和构建时间明显增加。目前 LLM 重排已能纠正顺序，暂不动。
- **查询改写 + 重排对 12 个 chunk 的小知识库偏重**：等真实内容量上来后再评估。

---

## 8. 容器化（一键启动）

- 新增根目录 `docker-compose.yml`（项目名 `goeuroops`），把后端、前端、Redis、ChromaDB、Prometheus 合成一个 Compose 项目，可在 Docker Desktop 里直接启停。
- 前端 nginx（`docker/nginx.conf`）：`/api/` 转发到 `backend:8000`，关闭代理缓冲（SSE）；额外转发 `/openapi.json`，否则 `/api/docs` 的 Swagger 页面拿不到接口定义而空白。
- 新增根目录 `.env.example` / `.gitignore`；删除已无用的 `frontend/docker-compose.yml`、`frontend/docker/nginx-gateway.conf`。

---

## 9. 整理成 GitHub 仓库 `GoEuroOps`

在原项目之外整理出这份独立的仓库（全新 `git init`，不带原有提交历史）。

- **改名**：项目正式命名为 `GoEuroOps 留学业务智能运营中枢`；`EchoMind/` → `backend/`，`EchoMindFrontend/` → `frontend/`；文本中的 `EchoMind` → `GoEuroOps`；环境变量前缀 `ECHOMIND_*` → `GOEUROOPS_*`；Docker 容器 / 网络 / npm 包名同步替换；banner 与 FastAPI 标题的副标题改为"留学业务智能运营中枢"。改名过程中修了两处被批量替换弄坏的相对路径（compose 构建上下文、README）。
- **刻意没有上传的内容**：`.env`（真实 key）、`.idea/`、`data/chroma/chroma.sqlite3`，以及与项目无关的个人材料。提交前扫描过文件内容，没有密钥。
- 新增顶层 `README.md`、`backend/.env.example`。

---

## 10. 遗留问题 / 待确认（v3 处理情况）

1. ~~RAG 缺少按意图门控的逻辑~~：**已在 v3 实现**，见 11.4。
2. ~~评测的"优化建议"总是"完整性偏低"~~：**已在 v3 修复**，见 11.7。根因不只是写死的规则，还有 Judge 用通用客服口径打分："正确地转交顾问"会被当成"没有完整解决问题"而扣分。
3. ~~文档已过时~~：**已在 v3 更新** 根目录 README、`backend/README.md`、`skills/README.md`，新增 `wiki/Skills与意图门控RAG.md`；wiki 中的旧文档加了版本说明，内容保留作为演进记录。
4. `backend/docker-compose.yml`（含 nginx）仍是"只启动后端"的旧入口，与根目录的新 compose 不要同时启动，会抢端口。**仍然存在。**

## 11. v3：工业级 Skills + 全系统工作室化（2026-09-23）

### 11.1 业务目录：单一数据源（`backend/business/`）

- `catalog.yaml` 定义以下内容，加载时用 pydantic 校验，写错会在启动时直接报错：
  - 工作室信息
  - 14 项服务：SKU、价格、包含与不包含、交付物、轮次、时效
  - 优惠规则：早鸟 9 折、团报 95 折、老带新减 ¥300，叠加后不低于标价 85%
  - 付款：套餐 30% 定金
  - 退款、发票、预约流程
- `countries.yaml`：五国的申请平台、时间窗口、申请费、学费、语言要求、代表项目、居留许可、奖学金、官网链接，并标注"待顾问核实"。
- `pricing.py`：报价和退款由代码确定性计算，模型只负责解释结果。
- `lead_store.py`：线索和交接单存在 Redis，Redis 不可用时降级为内存存储。登记前校验联系方式格式，拒绝证件号，日志中联系方式脱敏。
- `docker-compose.yml` 把 `business/` 挂载进容器，改价格后重启后端即可生效。

价目为草案（人民币、小工作室中档），需要创始人确认后再对外使用。

### 11.2 全系统工作室化

- **意图**：
  - `order_status` → `service_progress`
  - `logistics` → `booking`
  - `account_security` → `data_privacy`
  - 退款、发票、付款类意图的语义改为工作室费用场景。
- **实体**：新增合同号、入学季、语言分数、服务名，删除订单号。
- **修复意图识别的一个老 bug**：few-shot 示例一直计算了，但从未拼进 prompt。
- **关键词匹配**：纯英文关键词改为按单词边界匹配。
- **四个 Agent 重写为工作室角色**：
  - 前台接待
  - 留学咨询：新增报价和线索登记
  - 费用售后：新增退款估算
  - 转顾问：写交接单，回复中说明响应时效和时差
- **工具**：
  - `get_studio_profile`、`lookup_service_offering`、`quote_service_bundle`、`create_consultation_lead`
  - `check_payment_fields`、`calculate_refund`、`get_payment_policy`
  - `create_handoff_summary`（写入线索库）
- **修复**：成功路径上工具轨迹（tool_traces）丢失的问题。
- **复合问题检测**：只看留学主题词。问退款时顺口提到"全程陪跑"，不再触发咨询 Agent 并行协作；实测这类请求的耗时从 12s 降到 6.5s。

### 11.3 Skills v2（`core/skill_loader.py`）

- **目录规范**：`SKILL.md`、`references/`（渐进式披露，由 `read_skill_reference` 按需读取，只能读取本轮已命中的 Skill）、`evals/cases.json`。
- **front matter**：改为 YAML，新增 version、owner、intents、examples、mode、priority；加载时做 schema 校验，不合格的 Skill 被跳过并报告原因。
- **多信号路由**：意图绑定（0.5）、意图大类（0.25）、关键词（按单词边界匹配）、语义样例（字符 n-gram）、会话粘性、常驻，阈值 0.35，控制总预算。
- **可观测**：
  - 每次回答带 `skills_applied`，同时写入 trace、SSE 和 Prometheus 指标 `goeuroops_skill_hits_total`。
  - 统计命中次数，记录内容哈希。
  - 新增接口 `/skills/match`（命中测试）和 `/skills/evals`（命中回归）。
- **五个 Skill**：`studio_brand_voice`（常驻）、`general_reception`、`study_consulting`、`service_sales`、`billing_support`，共 9 份 references、29 条命中用例，准确率 100%。

### 11.4 意图门控 RAG（`core/rag_gate.py`）

- **三种模式**：prefetch（知识类意图预取并注入）、on_demand、off（不检索，并移除检索工具）；意图置信度低于 0.6 时退回 on_demand。
- **投机预取**：与意图识别并行，对三个 domain 做纯向量召回，意图确定后只采用主 Agent 所在 domain 的结果。
- **实测对照**（3 个知识类问题，两组都命中意图缓存）：门控组平均 **6.8s**、0 次工具调用；模型自行检索组平均 **10.1s**、1 次工具调用，**每次节省约 3.2s**。样本较少，只作为方向性结论。

### 11.5 知识库

- **中文向量模型**：默认英文模型 MiniLM 换成 **bge-small-zh-v1.5**（fastembed / ONNX，镜像构建时预下载）。
- **混合打分**：0.75 × 向量相似度 + 0.25 × 字面覆盖度；切片前加上【标题】再向量化。
- **自动迁移**：按向量模型区分 collection，首次启动时自动迁移用户导入的文档。
- **种子版本化**：种子由业务目录和 `business/articles/*.md` 生成，共 31 篇，带内容哈希版本号；内容变化时自动重建，并清理旧版的电商示例文档。
- 重排结果带上 relevance 分数。

### 11.6 前端

- 新增「线索」页：按状态和类型筛选、改状态、写备注；联系方式默认遮挡；需要时可以填写 Admin Token。
- 新增「服务价目」页：服务卡片、优惠和退款政策、五国资料。
- 新增「Skills」页：Skill 卡片显示版本、命中次数和 references；支持命中测试（各信号得分条）和命中回归。
- 对话页：
  - 每条回复显示 RAG 门控结果、命中的 Skill 及得分、调用的工具。
  - 快捷问题换成 6 个工作室场景。
- 知识库页：显示 domain 和来源分布，导入时可以选择归属。
- 评测页：
  - 新增门控对照开关、对话用例明细（含评审意见）。
  - 修复得分条宽度只有实际值 1/10 的 bug。

### 11.7 评测

- **Judge 按场景校准**：每个用例可以写期望行为；新增"边界合规"维度；评审会附上一句扣分原因。
- **确定性检查**：用例可以写 `expected_tools`，检查是否调用了期望的工具。
- **优化建议**：只有某个维度的均分低于阈值，或超过 30% 的样本低于阈值时才给出，并附上最差样本；Judge 调用失败的样本单独报告，不计入建议。
- **新指标**：Skill 路由准确率、工具调用准确率。
- **内置用例**：全部换成工作室场景，包括 15 条意图用例和 8 个对话用例。
- **实测结果**：通过率 13/13；意图准确率 100%，Skill 路由 100%，工具调用 100%；相关性 0.975、准确性 0.925、完整性 0.935（此前稳定在 0.65–0.69）、有用性 0.94、边界合规 1.0。

### 11.8 测试

测试从 11 条增加到 95 条：`test_business`、`test_skill_loader`（含所有 Skill 的回归用例参数化）、`test_rag_gate`（含"预取与意图识别并行"的时序测试）、`test_evaluator`，以及更新后的 `test_agent_orchestrator`。
