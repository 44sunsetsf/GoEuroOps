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

## 10. 遗留问题 / 待确认

1. **RAG 缺少按意图门控的逻辑**：第 7 节删掉的 `_should_use_knowledge` 是从未接线的死代码，现在代码里没有"按意图决定是否检索"的门控，RAG 是否触发完全由 Agent 通过工具调用自行判断。若需要门控，应把它真正接入主链路。
2. **评测的"优化建议"总是"完整性偏低"**：建议来自 `evaluation/evaluator.py` 的 `_recommendations()` 里写死的规则——`completeness` 均值低于 0.75 就固定输出这句话。当前实测完整性均值约 0.65–0.69，所以每次都会出现。这是规则触发，不是发现了新问题；要么调高 Agent 回复完整度，要么调整阈值 / 改成按低分用例动态生成建议。**尚未处理。**
3. **文档已过时**：`backend/README.md`、`backend/wiki/` 仍是"通用智能客服"的旧定位，尚未更新。
4. `backend/docker-compose.yml`（含 nginx）仍是"只启动后端"的旧入口，与根目录的新 compose 不要同时启动，会抢端口。
