"""轻量本地文本向量，不依赖外部 Embedding 服务。

两种表示：
  - hashed_ngram_embedding：定长哈希向量，意图识别三路融合里的 Embedding 分支使用；
  - ngram_profile + sparse_cosine：稀疏字符 n-gram 计数，Skill 语义触发使用。
    中文按 2/3 字切片、英文按单词切分，比哈希向量更稳定，也便于解释命中原因。
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Dict, Iterable, List

_ASCII_WORD = re.compile(r"[a-z][a-z0-9+#.\-]*")
_CJK_RUN = re.compile(r"[一-鿿]+")


def cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def hashed_ngram_embedding(text: str, dims: int = 256) -> List[float]:
    """稳定的字符 1–3 gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
    normalized = (text or "").lower().strip()
    vec = [0.0] * dims
    tokens = set()
    for n in (1, 2, 3):
        if len(normalized) >= n:
            tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
    if not tokens:
        tokens.add(normalized)

    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    return vec


def ngram_profile(text: str) -> Counter:
    """中文 2/3-gram + 英文单词的稀疏计数向量。"""
    lowered = (text or "").lower()
    grams: Counter = Counter()
    for run in _CJK_RUN.findall(lowered):
        for n in (2, 3):
            grams.update(run[i:i + n] for i in range(len(run) - n + 1))
        if len(run) == 1:
            grams[run] += 1
    grams.update(_ASCII_WORD.findall(lowered))
    return grams


def sparse_cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) < len(b) else (b, a)
    dot = sum(v * large.get(k, 0) for k, v in small.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def best_similarity(text: str, candidates: Iterable[Counter]) -> Dict[str, float]:
    """返回与候选样例的最高相似度及其下标，候选为空时为 0。"""
    profile = ngram_profile(text)
    best, best_idx = 0.0, -1
    for idx, cand in enumerate(candidates):
        sim = sparse_cosine(profile, cand)
        if sim > best:
            best, best_idx = sim, idx
    return {"similarity": best, "index": best_idx}
