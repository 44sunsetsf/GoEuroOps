# LoCoMo 记忆对比（amem-final-english）

- 第 0 段：19 个会话、419 句、40 题；第 1 段：19 个会话、369 句、30 题；共 70 题（每类每段最多 10）；模型 `deepseek-v4-flash`；单次运行
- 写入时的模型调用：amem-final-english 152 次 / 271595 token

| 条件 | 总体 | multi_hop | temporal | open_domain | single_hop | 提示词中记忆平均字数 | 证据会话全部被取回 | 至少取回一个 |
|---|---|---|---|---|---|---|---|---|
| amem-final-english | 47%（33/70） | 35% | 50% | 40% | 60% | 2637 | 63% | 79% |
