"""知识库向量模型。

ChromaDB 默认的 all-MiniLM-L6-v2 是英文模型，对中文几乎没有区分度
（实测"瑞典申请什么时候截止"召回的是服务价目）。这里换成 BAAI/bge-small-zh-v1.5：
  - fastembed + ONNX 推理，不依赖 torch，模型约 90MB，镜像构建时预下载；
  - 文档用 passage 编码，查询用带检索指令的 query 编码（bge 推荐用法）；
  - 模型加载失败时退回 ChromaDB 默认模型，保证服务可用。

通过 GOEUROOPS_EMBEDDING_MODEL 切换模型，设为 "default" 使用 ChromaDB 默认模型。
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


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


def load_embedding_function() -> Optional[FastEmbedFunction]:
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
