"""知识库向量模型。

ChromaDB 默认的 all-MiniLM-L6-v2 是英文模型，对中文几乎没有区分度
（实测"瑞典申请什么时候截止"召回的是服务价目）。这里换成 BAAI/bge-small-zh-v1.5：
  - fastembed + ONNX 推理，不依赖 torch，模型约 90MB，镜像构建时预下载；
  - 文档用 passage 编码，查询用带检索指令的 query 编码（bge 推荐用法）；
  - 模型加载失败时退回 ChromaDB 默认模型，保证服务可用。

通过 GOEUROOPS_EMBEDDING_MODEL 切换模型，设为 "default" 使用 ChromaDB 默认模型。

小内存服务器上可以改用远端向量 API（OpenAI 兼容的 /embeddings，例如硅基流动）：
设置 GOEUROOPS_EMBEDDING_PROVIDER=api 和 GOEUROOPS_EMBEDDING_API_KEY，默认模型为同系列的
BAAI/bge-large-zh-v1.5。进程里不再加载 ONNX 模型，常驻内存少 100MB 以上，启动也不再占满 CPU。
注意：意图识别的基准数字（380 条）是用本地 bge-small-zh 测的，换成 API 后需要重跑。
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_API_MODEL = "BAAI/bge-large-zh-v1.5"
DEFAULT_API_BASE_URL = "https://api.siliconflow.cn/v1"
# bge 中文模型推荐的检索查询指令（fastembed 的 query_embed 用的也是它）
BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class FastEmbedFunction:
    """同时满足 ChromaDB EmbeddingFunction 协议和查询编码需求。"""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name, cache_dir=cache_dir)

    # ChromaDB 会用它来编码写入的文档。0.5.x 的协议要求返回 numpy 数组列表
    # （HttpClient 发请求前会逐个调用 .tolist()）。
    def __call__(self, input: List[str]) -> List[Any]:  # noqa: A002 - ChromaDB 协议参数名
        return list(self._model.passage_embed(list(input)))

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self._model.query_embed([text]))).tolist()

    @property
    def collection_suffix(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.model_name.lower()).strip("_")


class ApiEmbedFunction:
    """远端向量 API，接口与 FastEmbedFunction 一致。查询向量带进程内缓存：
    意图识别和知识库检索经常编码同一句用户消息，缓存后只请求一次。"""

    _BATCH = 32
    _CACHE_SIZE = 1024

    def __init__(self, model_name: str, api_key: str, base_url: str = DEFAULT_API_BASE_URL,
                 timeout: float = 15.0, transport: Any = None):
        import httpx

        self.model_name = model_name
        self._url = base_url.rstrip("/") + "/embeddings"
        self._http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}, transport=transport)
        self._query_prefix = BGE_ZH_QUERY_PREFIX if "bge" in model_name.lower() and "zh" in model_name.lower() else ""
        self._cache: "dict[str, List[float]]" = {}

    def _embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self._BATCH):
            resp = self._http.post(self._url, json={"model": self.model_name, "input": texts[i:i + self._BATCH]})
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        return out

    def __call__(self, input: List[str]) -> List[Any]:  # noqa: A002 - ChromaDB 协议参数名
        import numpy as np

        return [np.asarray(v, dtype=np.float32) for v in self._embed(list(input))]

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            for text, vec in zip(todo, self._embed([self._query_prefix + t for t in todo])):
                if len(self._cache) >= self._CACHE_SIZE:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[text] = vec
        return [self._cache[t] for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_queries([text])[0]

    @property
    def collection_suffix(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.model_name.lower()).strip("_")


def load_embedding_function() -> Optional[Any]:
    if os.getenv("GOEUROOPS_EMBEDDING_PROVIDER", "local").strip().lower() == "api":
        model = os.getenv("GOEUROOPS_EMBEDDING_MODEL", "").strip() or DEFAULT_API_MODEL
        key = os.getenv("GOEUROOPS_EMBEDDING_API_KEY", "").strip()
        if key:
            base_url = os.getenv("GOEUROOPS_EMBEDDING_BASE_URL", "").strip() or DEFAULT_API_BASE_URL
            logger.info("知识库向量模型（远端 API）: %s", model)
            return ApiEmbedFunction(model, key, base_url)
        logger.error("GOEUROOPS_EMBEDDING_PROVIDER=api 但没有设置 GOEUROOPS_EMBEDDING_API_KEY，改用本地模型")
    model = os.getenv("GOEUROOPS_EMBEDDING_MODEL", DEFAULT_MODEL).strip()
    if not model or model.lower() == "default":
        logger.info("知识库使用 ChromaDB 默认向量模型")
        return None
    try:
        fn = FastEmbedFunction(model, cache_dir=os.getenv("FASTEMBED_CACHE_PATH") or None)
        logger.info("知识库向量模型: %s", model)
        return fn
    except Exception as ex:
        logger.error("加载向量模型 %s 失败，退回 ChromaDB 默认模型（中文检索效果会明显变差）: %s", model, ex)
        return None


_shared: Optional[Any] = None
_shared_loaded = False


def get_shared_embedding_function() -> Optional[Any]:
    """进程内共享的向量模型：知识库和意图识别共用一份，避免重复加载约 90MB 的模型。"""
    global _shared, _shared_loaded
    if not _shared_loaded:
        _shared = load_embedding_function()
        _shared_loaded = True
    return _shared

