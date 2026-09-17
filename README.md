# GoEuroOps

一个多 Agent 智能运营系统：意图识别、RAG 知识检索、Redis + ChromaDB 分层记忆、动态 Skills 注入、在线监控与降级、LLM-as-Judge 端到端评测。

当前配置的示例场景是一个留学咨询工作室（「指北」）的运营助手：介绍工作室的选校/文书服务、回答瑞典/德国/荷兰/芬兰/丹麦英语授课计算机硕士的公开知识问题，并在需要个性化判断时引导用户预约人工顾问。架构本身与具体业务场景解耦——General/Billing/Escalation 三个 Agent 仍是通用客服场景，只有 Consulting 这一个 Agent 槽位被改造成了留学咨询领域，用来验证"换一个业务场景需要改多少代码"。

## 目录结构

```text
backend/    Python 后端：FastAPI + 多 Agent 编排 + RAG + 记忆 + 监控 + 评测
frontend/   Vue 前端：对话调试台 / 知识库管理 / 评测面板
```

两部分可以独立运行，也可以通过各自的 `docker-compose.yml` 一起启动。详细说明见各自目录下的 README：

- [backend/README.md](backend/README.md) — 后端架构、API、快速开始
- [frontend/README.md](frontend/README.md) — 前端本地运行、Docker 部署

更详细的技术文档（架构图、业务流程、重点代码解读）在 [backend/wiki](backend/wiki) 下。

## 快速开始

```bash
cd backend
cp .env.example .env   # 填入 ANTHROPIC_API_KEY（或兼容 Anthropic 协议的第三方模型，如 DeepSeek）
docker compose up -d --build
```

后端启动后访问 `http://localhost:8000/docs` 看 API，或者启动前端：

```bash
cd frontend
npm install
npm run dev
```

## 技术栈

FastAPI · Anthropic-compatible LLM API（可切换 DeepSeek 等第三方模型）· ChromaDB（RAG + 情景记忆）· Redis（工作记忆）· Prometheus（监控）· Vue 3 + Vite（前端）
