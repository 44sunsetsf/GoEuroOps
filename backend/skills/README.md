# GoEuroOps Skills 文档

GoEuroOps 启动时会从 `GOEUROOPS_SKILLS_DIR` 读取 Skills，并在匹配用户请求时注入到对应 Agent 的 system prompt。Skills 适合维护业务处理规范、客服话术、留学咨询答疑边界、账单审核边界、升级规则和禁止事项。

当前内置三类 Skills：

```text
skills/general_customer_service/SKILL.md  # 通用客服：接待、澄清、分流、投诉和转人工
skills/study_consulting/SKILL.md          # 留学咨询：公开知识问答、服务介绍和人工顾问引导
skills/billing_support/SKILL.md           # 账单服务：扣款、退款、发票、订阅和财务审核
```

## Skill 文件格式

推荐每个 Skill 使用独立目录，并将主文件命名为 `SKILL.md`：

```text
skills/<skill_name>/SKILL.md
```

文件顶部使用简单 front matter：

```markdown
---
name: 留学咨询前台答疑规范
description: 适用于 ConsultingAgent 的留学申请公开知识问答和升级处理规范
keywords: 留学,申请,硕士,选校,文书,雅思,托福,瑞典,德国,荷兰,芬兰,丹麦
agents: consulting
enabled: true
---
```

字段说明：

- `name`：Skill 展示名称，会出现在注入给模型的 prompt 中。
- `description`：简短说明，方便 `/skills` 接口排查。
- `keywords`：触发关键词，用户消息命中后才注入；多个关键词用英文逗号或中文逗号分隔均可。
- `agents`：适用 Agent，可填 `general`、`consulting`、`billing`，多个值用逗号分隔。
- `enabled`：是否启用，支持 `true/false`。

## 编写要求

- 重要规则放在文档前半部分，因为过长内容会按 prompt 预算截断。
- 一类 Skill 只描述一类职责，不要把技术、账单、通用客服规则混在一个文件里。
- 必须包含“角色定位”“处理流程”“升级条件”“禁止事项”等稳定章节。
- 对用户隐私、支付、密码、验证码、API Key、Token 等敏感信息必须写明禁止收集或禁止公开。
- 对无法保证的事项使用保守措辞，例如“通常”“预计”“需要核验后确认”。
- 对需要人工、财务、二线技术处理的场景要明确写出升级条件。

## 热加载

修改 Skill 文件后，不需要重启服务，调用：

```bash
curl -X POST http://localhost:8000/skills/reload
```

查看加载结果和解析错误：

```bash
curl http://localhost:8000/skills
```
