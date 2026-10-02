"""Circuit breaker per model: stop sending traffic to a model that keeps failing."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_after_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> State:
        if self._opened_at is None:
            return State.CLOSED
        if self._clock() - self._opened_at >= self.reset_after_s:
            return State.HALF_OPEN
        return State.OPEN

    def allow(self) -> bool:
        state = self.state
        if state is State.CLOSED:
            return True
        if state is State.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True  # let exactly one probe through
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._probe_in_flight = False

    def record_failure(self) -> None:
        self._failures += 1
        if self._probe_in_flight or self._failures >= self.failure_threshold:
            self._opened_at = self._clock()
        self._probe_in_flight = False
