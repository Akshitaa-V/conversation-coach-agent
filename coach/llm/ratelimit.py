"""Async token bucket per vendor, so a burst never exceeds the vendor's rate limit.

Callers pass a priority (0 = most urgent). When requests queue, a waiting
higher-priority request always gets the next token, so live roleplay turns are
not stuck behind a burst of feedback or judge calls, while either kind can use
capacity the other leaves idle.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable


class TokenBucket:
    def __init__(
        self,
        rate_per_minute: int,
        burst: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self.rate = rate_per_minute / 60.0
        self.capacity = float(burst if burst is not None else max(1, rate_per_minute // 10))
        self.tokens = self.capacity
        self._clock = clock
        self._last = clock()
        self._queues: dict[int, deque[object]] = {}
        self.waited_s = 0.0  # total time callers spent throttled, for metrics

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
        self._last = now

    def _is_next(self, priority: int, ticket: object) -> bool:
        """FIFO within a priority, strict priority across priorities."""
        first = min(p for p, q in self._queues.items() if q)
        return first == priority and self._queues[priority][0] is ticket

    async def acquire(self, priority: int = 1) -> float:
        """Wait until a request may be sent. Returns the seconds spent waiting.

        The check-and-take below runs without an ``await`` in between, so it is
        atomic on the event loop and needs no lock.
        """
        waited = 0.0
        ticket = object()
        queue = self._queues.setdefault(priority, deque())
        queue.append(ticket)
        try:
            while True:
                self._refill()
                if self.tokens >= 1 and self._is_next(priority, ticket):
                    self.tokens -= 1
                    self.waited_s += waited
                    return waited
                delay = (1 - self.tokens) / self.rate if self.tokens < 1 else 0.001
                delay = max(delay, 0.001)
                waited += delay
                await asyncio.sleep(delay)
        finally:
            queue.remove(ticket)
