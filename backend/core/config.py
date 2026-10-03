"""集中放配置相关的公共部分：读取环境变量的工具函数和默认模型名。

各模块的开关仍然在用到的地方读取（就近、好找），但读取方式统一走这里：
配置写错时记一条警告并用默认值，不阻塞服务启动。
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# 没有设置 ANTHROPIC_MODEL 时的默认模型；各组件的构造函数默认值都引用这一个常量
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


def env_float(name: str, default: float) -> float:
    """读取可选浮点配置；没设置用默认值，写错记警告后用默认值。"""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("忽略非法浮点配置 %s=%r", name, raw)
        return default


def env_int(name: str, default: int) -> int:
    """读取可选整数配置；没设置用默认值，写错记警告后用默认值。"""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("忽略非法整数配置 %s=%r", name, raw)
        return default
