<p align="right"><a href="https://github.com/44sunsetsf/GoEuroOps"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/images/lang-zh-dark.svg"><img alt="EN | 中文 — 切换到英文" src="docs/images/lang-zh-light.svg" width="112"></picture></a></p>

# GoEuroOps 留学业务智能运营中枢

「指北 · Nordic CS Master Studio」的多 Agent 运营中枢。指北是由几位在瑞典读 CS 的留学生创办的咨询工作室，只做瑞典、德国、荷兰、芬兰、丹麦的英语授课计算机硕士申请。

AI 助手负责工作室的前台工作：

- 回答五国 CS 硕士的公开知识问题
- 介绍服务，并按公开规则**计算报价**
- 在用户同意后**登记咨询线索**
- 解释付款和退款政策，**估算可退金额**
- 遇到个性化选校、文书修改、录取判断这类需要顾问的请求时，交接给真人顾问

学生从学生端首页了解五国和服务，并和助手对话；创始人在工作室后台跟进线索、维护价目和 Skills、查看评测结果。

## 技术要点

| 模块 | 做法 |
|---|---|
| 意图识别 | LLM few-shot + 本地向量 + 关键词三路融合，15 个工作室业务意图 |
| 多 Agent 编排 | 前台接待 / 留学咨询 / 费用售后 / 转顾问，复合问题并行协作后合并 |
| 意图门控 RAG | 知识类意图与意图识别**并行**做投机预取，省掉一轮工具调用；问候、投诉、隐私不检索 |
| 知识库 | ChromaDB + bge-small-zh 中文向量（ONNX）+ 字面覆盖混合打分；种子由业务目录生成并带版本 |
| Skills v2 | 多信号路由（意图 / 关键词 / 语义样例 / 会话粘性）、渐进式披露、版本与命中统计、每个 Skill 自带回归用例 |
| 业务目录 | `business/catalog.yaml` 统一定义服务、价格、优惠、退款规则；报价和退款由代码确定性计算 |
| 评测 | 意图准确率、Skill 路由准确率、按场景校准的五维 LLM-as-Judge（含边界合规）、工具调用检查、门控 A/B |
| 观测 | 每次回答带 Skill 命中、门控决策和工具调用明细；Prometheus 指标 |

## 目录结构

```text
backend/    Python 后端：FastAPI + 多 Agent + RAG + Skills + 业务目录 + 评测
frontend/   Vue：学生端首页与对话（/），工作室后台（/studio：对话调试 / 线索 / 服务价目 / Skills / 知识库 / 评测）
docs/       变更记录
```

- [backend/README.md](backend/README.md)：后端架构、接口、配置
- [backend/skills/README.md](backend/skills/README.md)：Skill 编写规范
- [backend/wiki/Skills与意图门控RAG.md](backend/wiki/Skills与意图门控RAG.md)：核心设计说明
- [docs/CHANGELOG-2026-09.md](docs/CHANGELOG-2026-09.md)：变更记录
- [docs/ROADMAP-commercial.md](docs/ROADMAP-commercial.md)：商用化差距评估与优化计划

## 快速开始

在项目根目录一条命令启动全部服务（后端、前端、Redis、ChromaDB、Prometheus）：

```bash
cp .env.example .env   # 填入 ANTHROPIC_API_KEY（或兼容 Anthropic 协议的第三方模型，如 DeepSeek）
docker compose up -d --build
```

| 入口 | 地址 |
|---|---|
| 学生端 | http://localhost（端口可用 `.env` 里的 `FRONTEND_PORT` 修改） |
| 工作室后台 | http://localhost/studio |
| API 文档 | http://localhost/api/docs 或 http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |

首次构建后，可以直接在 Docker Desktop 的 Containers 页面对 `goeuroops` 项目点击启动或停止。

**哪些改动需要重建镜像**

| 改了什么 | 怎么生效 |
|---|---|
| `backend/skills/` | 已挂载进容器，改完调用 `POST /skills/reload` 或在工作室后台点「重新加载」 |
| `backend/business/`（价格、政策、五国资料） | 已挂载，改完执行 `docker compose restart backend`；知识库种子会按内容版本自动重建 |
| 其他 `backend/` 或 `frontend/` 代码 | 需要重新执行 `docker compose up -d --build` |

**本地开发前端（热更新）**：保持后端容器运行，然后执行：

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173，自动代理到 localhost:8000
```

**跑后端测试**：

```bash
cd backend && pip install -r requirements.txt pytest && python -m pytest -q
```

## 技术栈

FastAPI · Anthropic 兼容 LLM API（可切换 DeepSeek 等）· ChromaDB · fastembed（bge-small-zh）· Redis · Prometheus · Vue 3 + Vite
