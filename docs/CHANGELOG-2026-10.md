# 2026-10 改动记录

## 1. 工具层重构：统一网关、Pydantic 定义、金额护栏、只读 MCP 入口（2026-10-01）

### 问题
- 两套工具抽象并存：13 个 Agent 工具只有白名单和手写的简化校验，缓存、超时、熔断、降级只落在检索这一个工具上。
- 参数校验是手写的，不支持嵌套（报价的 `items` 数组要靠函数自己检查），返回值完全没有校验。
- "金额只能由工具算"只靠工具说明和提示词约束，线上没有拦截；trace 只在内存里保留最近 200 条。
- `mcp/` 目录并不是 MCP 协议，名字有误导，还占用了官方 SDK 的包名。

### 改动
| 改动 | 位置 |
|---|---|
| `mcp/` 改名 `retrieval/`，`MCPToolManager` 改名 `RetrievalManager` | `retrieval/` |
| 熔断器抽成共用模块 | `tooling/breaker.py` |
| 全部 Agent 工具的入参改用 Pydantic 定义；发给模型的 JSON Schema 由模型生成（内联 `$ref`、折叠 Optional） | `tooling/schemas.py`、`agents/tools.py` |
| 报价、退款的返回值模型检查业务不变量；不满足就不把结果交给模型 | `tooling/schemas.py` |
| 工具网关：每个工具声明超时、副作用类型、是否熔断、结果上限；同步函数放线程执行 | `tooling/gateway.py` |
| 结果过大时截断列表并标 `truncated`、`truncated_lists` | `tooling/gateway.py` |
| 金额护栏：流式按句放行，没有依据的金额所在句被替换；`GOEUROOPS_AMOUNT_GUARD` 三档 | `tooling/amount_guard.py`、`agents/agent_orchestrator.py` |
| 工具指标和 `/tools/stats` | `core/metrics.py`、`api/main.py` |
| 只读工具的 MCP 入口（FastMCP），依赖单独装 | `mcp_server.py`、`requirements-mcp.txt` |

### 验证
- 测试 140 → 165：网关（超时、熔断、异常隔离、返回值校验、截断、统计）、嵌套参数校验、
  金额护栏（抽取、替换、流式、放行有依据的金额）、Agent 端到端（编造总价被拦、工具算出的金额放行、
  非法参数回传给模型）、MCP 协议往返。
- 护栏关掉（`off`）后端到端的两个护栏测试会失败，说明测试确实在检查护栏。
- 金额依据集合每次请求多花约 0.26 ms。

### 没做
- 输入侧用 `tool_choice` 强制先调用报价工具：当前模型服务是 DeepSeek 的 Anthropic 兼容接口，对 `tool_choice` 的支持没有验证过。
- trace 落盘和审计、请求整体时间预算、`create_consultation_lead` 的幂等键：下一步。

## 2. 记忆模块按 A-Mem 思路重写，只保留一种记忆（2026-10-06）

### 问题
- 记忆叠了五层：Redis 工作记忆、会话摘要、情景记忆（摘要向量）、逐句索引、用户画像，再加一个可切换的 A-Mem 魔改笔记层（`GOEUROOPS_MEMORY_IMPL`），两套实现、两个开关，每层都要单独提炼或压缩。
- 记忆的向量用的是 ChromaDB 默认的英文 MiniLM，中文几乎没有区分度：压力题里问“我现在每年的预算是多少”，三句讲预算的话一句都没取回，取回的全是闲聊。
- ChromaDB 带过滤条件的查询偶尔报 “Cannot return the results in a contigious 2D array”，原来这时整次检索返回空，模型看不到任何记忆。
- 自建 30 题所有方案都是 100%，分不出好坏。

### 改动
| 改动 | 位置 |
|---|---|
| 只剩一种记忆：最近 14 条原文（Redis 滑动窗口，不调模型）+ A-Mem 笔记（ChromaDB） | `memory/amem.py` |
| 用户的每句陈述立刻存成原文笔记；同一句话再说只刷新时间 | `memory/amem.py` |
| 一次模型调用批量补全：关键词、标签、背景、事件时间、关键事实字段、链接、取代；时机是攒够 8 句、换了会话、或用户空闲 2 分钟 | `memory/amem.py` |
| 演化只追加：旧背景进 versions 并重算向量，被取代的笔记标“后来有更新” | `memory/amem.py` |
| 关键事实不再单独维护画像，从笔记的字段汇总：单值字段以最新为准、旧值留档，经历和偏好列出全部 | `memory/amem.py` |
| 向量模型和知识库共用 bge-small-zh，查询带检索指令编码 | `memory/amem.py` |
| 向量查询报错时，在这位用户的笔记里精确计算距离 | `memory/amem.py` |
| 后台任务保留引用；关闭服务时取消空闲计时、最多等 5 秒 | `memory/amem.py`、`api/routes/chat.py` |
| 删除旧实现和两个开关；旧代码在分支 `backup/memory-v2-and-amem-modded`、tag `memory-legacy-backup` | `memory/`、`docker-compose.yml`、`.env.example` |
| 新增 10 道压力题：闲聊陈述也会存成笔记（含干扰项）、多会话、反复修改、相对时间、多跳 | `evaluation/memory_benchmark.py` |

### 验证
每类都跑 3 次，模型 deepseek-v4-flash；“维护调用”是每次运行里记忆模块自己调模型的次数。

| | 旧 v2（原线上） | 旧 A-Mem 魔改 | **新版** |
|---|---|---|---|
| 基础 + 难题 30 题 | 100%（turn-index 版，80 次） | 100%（96 次） | **100%（33 次）** |
| 压力题 10 题（pass^3） | 93%（80%），140 次 | 97%（90%），167 次 | **100%（100%），22 次** |
| LoCoMo 70 题（单次，英文向量模型，三者相同） | 33% | 46%，写入 242 次 | **47%，写入 152 次** |
| LoCoMo 70 题（单次，线上的中文向量模型） | — | — | 41% |

为了确认每个改动都有用，还跑了去掉某一部分的对照（报告在 `evaluation/reports/`）：

| 对照 | 压力题 | 说明 |
|---|---|---|
| 只有笔记、英文向量模型 | 63% | 中文检索取不回；4 题因为查询报错整次返回空 |
| 加中文向量模型 | 83% | 剩下的是报错返回空和多跳 |
| 加单独的用户画像（旧做法） | 97% | 画像在弥补检索差，代价是每轮多一次提炼（103 次） |
| 新版：中文向量 + 报错降级 + 字段汇总 + 空闲补全 | 100% | 22 次 |

- LoCoMo 是英文数据，中文向量模型在上面吃亏（41%），所以和旧版比时三者都用英文模型（`--english-embedder`）；线上中文用户用中文模型。
- 测试：新增 `tests/test_amem.py`（15 个），全部后端测试 187 个通过，`tests/test_memory_fallback.py` 改成新接口；删掉旧实现的测试。
- 降级路径的测试用 numpy 数组模拟 ChromaDB 的返回值：第一版写了 `embeddings or []`，numpy 数组会报错，LoCoMo 评测里触发后才发现。

### 没做
- 旧记忆不迁移：线上是演示站，旧的 `episodic`、`turns`、`user_profile`、`amem_notes` 集合留在库里不再读。
- “删除我的资料”还没有真正删除笔记（ROADMAP P0 第 2 条）。
- 自建题已经封顶，下一步要加更长、更乱的真实对话。
