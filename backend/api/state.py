"""应用运行时用到的组件。

lifespan 启动时创建并挂到这里，各路由模块通过 services.xxx 使用；
测试里直接给对应属性赋假的替身即可，不用启动真实的模型、Redis 和 ChromaDB。
"""
import asyncio
from typing import Any, Optional


class Services:
    def __init__(self) -> None:
        self.orchestrator: Optional[Any] = None
        self.memory: Optional[Any] = None
        self.tool_manager: Optional[Any] = None
        self.monitor: Optional[Any] = None
        self.evaluator: Optional[Any] = None
        self.skill_manager: Optional[Any] = None
        self.lead_store: Optional[Any] = None
        self.quota: Optional[Any] = None              # 公开演示的每日对话上限
        self.guest_eval_quota: Optional[Any] = None   # 演示访客每天能跑几次评测
        self.guest_eval_lock = asyncio.Lock()         # 演示访客同一时间只跑一个评测


services = Services()
