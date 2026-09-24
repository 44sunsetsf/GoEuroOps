<p align="right"><a href="README.zh-CN.md"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/images/lang-dark.svg"><img alt="EN | 中文 — switch to Chinese" src="docs/images/lang-light.svg" width="112"></picture></a></p>

# GoEuroOps — an AI operations hub for a study-abroad studio

The multi-agent operations hub behind **Zhibei (指北) · Nordic CS Master Studio**, a consultancy started by a few students doing CS in Sweden. The studio focuses on one thing: applications to English-taught CS master's programmes in Sweden, Germany, the Netherlands, Finland and Denmark.

The AI assistant runs the studio's front desk:

- answers public questions about CS master's programmes in the five countries
- introduces the services and **calculates quotes** from the published pricing rules
- **records consultation leads** once the user agrees
- explains payment and refund policies and **estimates refundable amounts**
- hands over to a human advisor when a request needs one — personalised school selection, essay editing, admission judgements

Students learn about the five countries and the services on the public site and chat with the assistant there; the founders use the studio console to follow up on leads, maintain prices and Skills, and review evaluation results.

## Highlights

| Module | Approach |
|---|---|
| Intent recognition | Three signals fused — LLM few-shot, local embeddings and keywords — across 15 studio-specific intents |
| Multi-agent orchestration | Front desk / study consulting / billing & after-sales / human handover; compound questions are handled in parallel and merged |
| Intent-gated RAG | Knowledge intents trigger a speculative prefetch **in parallel** with intent recognition, saving a tool-call round trip; greetings, complaints and privacy questions skip retrieval |
| Knowledge base | ChromaDB + bge-small-zh Chinese embeddings (ONNX) with hybrid lexical-coverage scoring; seed documents are generated from the business catalog and versioned |
| Skills v2 | Multi-signal routing (intent / keywords / semantic examples / conversation stickiness), progressive disclosure, versions and hit statistics; every Skill ships its own regression cases |
| Business catalog | `business/catalog.yaml` defines services, prices, discounts and refund rules in one place; quotes and refunds are computed deterministically in code |
| Evaluation | Intent accuracy, Skill-routing accuracy, a scenario-calibrated five-dimension LLM-as-Judge (including boundary compliance), tool-call checks, gating A/B |
| Observability | Every answer carries its Skill hits, gating decision and tool-call details; Prometheus metrics |

## Repository layout

```text
backend/    Python backend: FastAPI + multi-agent + RAG + Skills + business catalog + evaluation
frontend/   Vue: student site and chat (/), studio console (/studio: chat debugging / leads / pricing / Skills / knowledge base / evaluation)
docs/       change log and roadmap
```

- [backend/README.md](backend/README.md): backend architecture, API, configuration
- [backend/skills/README.md](backend/skills/README.md): how to write a Skill
- [backend/wiki/Skills与意图门控RAG.md](backend/wiki/Skills与意图门控RAG.md): core design notes
- [docs/CHANGELOG-2026-09.md](docs/CHANGELOG-2026-09.md): change log
- [docs/ROADMAP-commercial.md](docs/ROADMAP-commercial.md): gap analysis and plan towards production use

The documents above are written in Chinese.

To put it on a public server with HTTPS, rate limiting and a password-protected console, see [deploy/DEPLOY.md](deploy/DEPLOY.md) (in Chinese).

## Quick start

Start every service (backend, frontend, Redis, ChromaDB, Prometheus) from the project root with one command:

```bash
cp .env.example .env   # set ANTHROPIC_API_KEY (or any Anthropic-compatible provider, e.g. DeepSeek)
docker compose up -d --build
```

| Entry | URL |
|---|---|
| Student site | http://localhost (change the port with `FRONTEND_PORT` in `.env`) |
| Studio console | http://localhost/studio |
| API docs | http://localhost/api/docs or http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |

After the first build you can start and stop the `goeuroops` project from the Containers page in Docker Desktop.

**What needs a rebuild**

| What changed | How to apply it |
|---|---|
| `backend/skills/` | Mounted into the container; call `POST /skills/reload` or click "Reload" in the studio console |
| `backend/business/` (prices, policies, country data) | Mounted; run `docker compose restart backend` — the knowledge-base seed is rebuilt automatically when its content version changes |
| Any other `backend/` or `frontend/` code | Run `docker compose up -d --build` again |

**Frontend development with hot reload** — keep the backend container running, then:

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173, proxied to localhost:8000
```

**Backend tests**:

```bash
cd backend && pip install -r requirements.txt pytest && python -m pytest -q
```

## Tech stack

FastAPI · Anthropic-compatible LLM API (DeepSeek and others work too) · ChromaDB · fastembed (bge-small-zh) · Redis · Prometheus · Vue 3 + Vite
