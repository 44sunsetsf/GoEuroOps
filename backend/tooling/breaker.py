"""三态熔断器：Agent 工具网关和检索管理器共用。"""
from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED    = "closed"     # 正常
    OPEN      = "open"       # 熔断，拒绝请求
    HALF_OPEN = "half_open"  # 探测恢复


class CircuitBreaker:
    """
    三态熔断器：CLOSED → OPEN → HALF_OPEN → CLOSED

    连续失败 failure_threshold 次后打开；
    打开 recovery_s 秒后进入 HALF_OPEN，只放行一个探测请求；
    探测成功则关闭，失败则重新打开。

    用法：ticket = breaker.acquire()，None 表示拒绝；调用结束后把 ticket 交回
    record_success / record_failure。每次状态切换都换一代 ticket，旧 ticket 的结果直接忽略：
    熔断前发出、熔断后才返回的慢请求不能把熔断器提前关上，半开期间也只有探测请求的结果算数。
    """

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold   = failure_threshold
        self.recovery_s  = recovery_s
        self.state       = CircuitState.CLOSED
        self.fail_count  = 0
        self.opened_at:  Optional[float] = None
        self._generation = 0
        self._probe_at:  Optional[float] = None

    def _switch(self, state: CircuitState) -> None:
        self.state = state
        self._generation += 1

    def acquire(self) -> Optional[int]:
        """请求放行。放行时返回本次调用的 ticket，拒绝时返回 None。"""
        now = time.monotonic()
        if self.state == CircuitState.CLOSED:
            return self._generation
        if self.state == CircuitState.OPEN:
            if now - self.opened_at < self.recovery_s:  # type: ignore[operator]
                return None
            self._switch(CircuitState.HALF_OPEN)
            self._probe_at = now
            return self._generation
        # HALF_OPEN：探测请求还在路上就拒绝；探测被取消、迟迟不回时，过了 recovery_s 再放一个新的
        if self._probe_at is not None and now - self._probe_at < self.recovery_s:
            return None
        self._switch(CircuitState.HALF_OPEN)
        self._probe_at = now
        return self._generation

    def record_success(self, ticket: Optional[int] = None) -> None:
        if ticket is not None and ticket != self._generation:
            return
        self.fail_count = 0
        if self.state != CircuitState.CLOSED:
            self._switch(CircuitState.CLOSED)
            self._probe_at = None
            logger.info("熔断器恢复")

    def record_failure(self, ticket: Optional[int] = None) -> None:
        if ticket is not None and ticket != self._generation:
            return
        self.fail_count += 1
        if self.state == CircuitState.HALF_OPEN or self.fail_count >= self.threshold:
            self._switch(CircuitState.OPEN)
            self.opened_at = time.monotonic()
            self._probe_at = None
            logger.warning(f"熔断器打开（连续失败 {self.fail_count} 次）")
