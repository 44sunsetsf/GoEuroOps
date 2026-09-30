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
