"""校准 wiki 选页兜底的阈值：LLM 选页失败时，用问题去匹配目录里每页的"常见问法"。

两种兜底都在 retrieval_v1（106 条）上扫阈值，看注入命中率和无答案误注入率：
  embedding   向量余弦（问题和问法用同一种编码，对称匹配），线上用 bge-large API
  ngram       字符 n-gram 余弦（不依赖任何网络，最后一层兜底）

用法（在 backend/ 目录下；线上的向量模型走 API，key 只在服务器上，所以在线上容器里跑）：
    python -m evaluation.wiki_fallback_calibrate
    docker cp evaluation/wiki_fallback_calibrate.py goeuroops-backend-1:/app/evaluation/ &&
    docker exec -w /app goeuroops-backend-1 python -m evaluation.wiki_fallback_calibrate

注意：阈值在评测集上选，会有轻微的过拟合；这里只用于兜底路径，正常路径是 LLM 选页。
"""
from __future__ import annotations

import time
from typing import Dict, List

from core.rag_gate import retrieval_query
from core.text_embedding import ngram_profile, sparse_cosine
from evaluation.retrieval_benchmark import history_of, load_dataset, matches
from retrieval.embeddings import get_shared_embedding_function
from retrieval.wiki import build_pages


def _unit(v: List[float]) -> List[float]:
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def sweep(name: str, scores: List[Dict[str, float]], rows: List[dict], sources: Dict[str, List[str]],
          thresholds: List[float], max_pages: int = 2) -> None:
    n_ans = sum(1 for r in rows if r["expected"])
    n_no = len(rows) - n_ans
    print(f"\n{name}")
    for t in thresholds:
        hit = false = 0
        for r, s in zip(rows, scores):
            sel = [pid for pid, v in sorted(s.items(), key=lambda x: -x[1])[:max_pages] if v >= t]
            if r["expected"]:
                hit += any(matches(src, r["expected"]) for pid in sel for src in sources[pid])
            else:
                false += bool(sel)
        print(f"  阈值 {t:.2f}: 命中 {hit / n_ans:.1%}  无答案误注入 {false / max(1, n_no):.1%}")


def main() -> None:
    pages = build_pages()
    sources = {p.id: p.sources for p in pages}
    phrases = [(p.id, t) for p in pages for t in [p.title, *p.questions] if t.strip()]
    rows = load_dataset("retrieval_v1")
    queries = [retrieval_query(r["query"], history_of(r)) for r in rows]

    ngram = [ngram_profile(t) for _, t in phrases]
    ng_scores = []
    for q in queries:
        qp, best = ngram_profile(q), {}
        for (pid, _), prof in zip(phrases, ngram):
            best[pid] = max(best.get(pid, 0.0), sparse_cosine(qp, prof))
        ng_scores.append(best)
    sweep("ngram", ng_scores, rows, sources, [0.2, 0.25, 0.3, 0.35, 0.4])

    fn = get_shared_embedding_function()
    if fn is None:
        print("\n没有可用的向量模型，跳过 embedding")
        return
    t0 = time.monotonic()
    pv = [_unit(list(map(float, v))) for v in fn([t for _, t in phrases])]
    t1 = time.monotonic()
    qv = [_unit(list(map(float, v))) for v in fn(queries)]
    t2 = time.monotonic()
    emb_scores = []
    for q in qv:
        best: Dict[str, float] = {}
        for (pid, _), p in zip(phrases, pv):
            best[pid] = max(best.get(pid, -1.0), sum(a * b for a, b in zip(q, p)))
        emb_scores.append(best)
    model = getattr(fn, "model_name", type(fn).__name__)
    print(f"\nembedding（{model}）：{len(phrases)} 条问法编码 {1000 * (t1 - t0):.0f} ms，"
          f"{len(queries)} 条问题编码 {1000 * (t2 - t1):.0f} ms")
    sweep(f"embedding {model}", emb_scores, rows, sources, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])


if __name__ == "__main__":
    main()
