# Skills v2 与意图门控 RAG 设计说明

本文说明 v3 版本中两块核心设计：Skills 框架如何挑选并注入业务规范，以及知识库检索如何按意图门控。

## 一、Skills v2

### 1. 要解决的问题

v2 之前的 Skill 只有"关键词子串命中 → 整篇正文塞进 system prompt"，存在五个问题：

- **误命中与漏命中**：`cs` 会命中 `docs`；用户说"想去北欧读研"时，因为没出现关键词表里的词，Skill 不会被注入。
- **没有校验**：front matter 写错了静默失效；README 写支持中文逗号，实际只按英文逗号切分。
- **没有分层**：所有细节都常驻 prompt，挤占上下文。
- **不可观测**：看不出某次回答用了哪个 Skill、哪个版本、为什么命中。
- **不可回归**：改一个关键词可能让别的场景误命中，但没有测试能发现。

### 2. 结构（对齐 Anthropic Agent Skills）

```text
skills/<id>/SKILL.md          核心规则：命中即注入
skills/<id>/references/*.md   详细资料：只列目录，模型用 read_skill_reference 按需读取（渐进式披露）
skills/<id>/evals/cases.json  命中回归用例
```

### 3. 路由：为什么不让模型自己挑

官方 Skills 的做法是：模型先看所有 Skill 的 description，再自己决定加载哪个。本项目没有这样做，原因有两点：

- **延迟**：模型自己挑需要多一轮 LLM 往返，而这里的 Agent 是固定的业务角色，对延迟敏感。
- **信号现成**：请求到达 Agent 时，已经有了三路融合的意图识别结果，这是比 description 更强的信号。

所以 `SkillRouter` 用确定性打分决定是否注入，阈值为 0.35：

| 信号 | 分值 | 设计理由 |
|---|---|---|
| 具体意图 ∈ `intents` | 0.50 | 复用已算好的意图，零额外成本，单独即可命中 |
| 仅意图大类 ∈ `intents` | 0.25 | 同组的其他 Skill 不应被顺带注入（实测发现 service_inquiry 会把 study_consulting 带进来） |
| 关键词 | 0.35 起，最多 0.45 | 纯英文关键词按单词边界匹配 |
| 语义样例 | 最多 0.40 | 字符 2/3-gram 余弦相似度，覆盖关键词表之外的说法，也能兜住意图识别出错的情况 |
| 会话粘性 | 补足到阈值 | 上一轮单独命中时，本轮追问继承命中，只回看一轮 |
| `mode: always` | 1.0 | 品牌语气这类必须常驻的规范 |

命中后按 always → priority → 得分排序，并控制总预算（默认 6000 字）。

### 4. 渐进式披露与安全

`read_skill_reference` 在每次 Agent 调用时，以闭包形式临时构建：

- 只允许读取**本轮已命中** Skill 的已登记文件名。
- 文件名必须满足 `^[A-Za-z0-9_\-.]+\.md$`，并且必须在登记列表里，所以不存在路径穿越。

### 5. 可观测与回归

- **可观测**：每次响应都带 `skills_applied`（id、版本、得分、各信号和原因），同时写入 trace、SSE `done` 事件，以及 Prometheus 指标 `goeuroops_skill_hits_total`。`SkillManager` 维护命中次数和最近命中时间，热加载后不清零；内容哈希用来判断热加载后内容是否真的变了。
- **回归**：`evals/cases.json` 由 `tests/test_skill_loader.py` 参数化执行（CI），也可以通过 `/skills/evals` 执行（线上），`/eval/run` 会把结果报告为 `skill_routing_accuracy`。开发这套机制时，回归用例已经发现过两个真实问题：一是关键词缺口（"顾问三天都没回我"），二是会话粘性的系数设计让粘性实际不生效。

## 二、意图门控 RAG

### 1. 要解决的问题

原先的 `_should_use_knowledge` 是写了但没有被调用的死代码，RAG 完全依赖模型自己调用检索工具。这带来三个问题：

- 知识类问题要多一轮 LLM 往返：模型先返回 tool_use，拿到结果后再回答。
- 问候、投诉这类消息也能看到检索工具，偶尔被无效调用。
- 设计上说的"意图门控 RAG"实际并没有生效。

### 2. 策略表（`core/rag_gate.py`）

| 模式 | 意图 | 行为 |
|---|---|---|
| prefetch | study_consult、application_process、service_inquiry、booking、refund、invoice、payment_issue、billing | 预取 top-3，作为「知识库上下文」注入并标注来源；检索工具保留，模型可以补查 |
| on_demand | query、request、service_progress、account、other | 维持原行为，由模型决定是否检索 |
| off | greeting、feedback、complaint、escalation、human_handoff、data_privacy | 不检索，并从本轮工具列表中移除检索工具 |

意图置信度低于 0.6 时，prefetch 和 off 都降级为 on_demand，避免意图识别出错导致该查的知识没有查。

### 3. 投机预取：不增加延迟

```text
t=0   ┬─ 意图识别（一次 LLM 调用，约 1–2s）
      └─ 投机预取：consulting / billing / general 三个 domain 各做一次纯向量召回（约 10ms 级）
t=意图完成 → 路由确定主 Agent 的 domain
      → prefetch：取该 domain 的结果，过滤掉混合分数 < 0.4 的片段，其余 domain 的任务取消
      → on_demand / off：全部取消
```

预取走 `MCPToolManager.search_fast`，不做 LLM 改写和重排，但仍然经过缓存、熔断和降级。模型主动调用的 `search_knowledge_base` 仍然走完整链路：改写 → 并行召回 → 去重 → LLM 重排。

### 4. 检索质量：中文向量模型 + 混合打分

ChromaDB 默认的 all-MiniLM-L6-v2 是英文模型。实测问"瑞典申请什么时候截止"，召回的是服务价目文档。门控会把预取结果直接注入 prompt，所以检索质量必须先解决：

- 向量模型换成 **bge-small-zh-v1.5**：用 fastembed 做 ONNX 推理，不依赖 torch，模型约 90MB，镜像构建时预下载，单次查询约 1ms。
- 按向量模型区分 collection。首次启动时自动从旧 collection 迁移用户导入的文档，然后删除旧 collection。
- 切片前加上「【标题】」再向量化，弥补切片后正文缺少主题词的问题。
- 混合打分：0.75 × 余弦相似度 + 0.25 × 字面覆盖度。字面覆盖度用来兜住 APS、TUM 这类夹在中文里的专有名词。

实测相关片段的分数约为 0.48–0.77，不相关的（如"今天天气怎么样"）约为 0.24，因此预取阈值取 0.4。

### 5. 观测与对照实验

- **观测**：每次响应带 `rag_gate: {mode, reason, prefetched, hits, wait_ms}`，Prometheus 指标为 `goeuroops_rag_gate_total{mode}`。
- **对照实验**：`POST /eval/run {"compare_rag_gate": true}` 会对知识类用例分别跑门控组和"模型自行检索"组，对比延迟和检索调用次数。开始前先预热意图缓存，让两组都命中缓存，这样差异只来自检索策略本身。
