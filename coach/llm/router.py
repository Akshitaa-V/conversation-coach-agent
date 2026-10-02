"""Model router: one place for retries, fallbacks, timeouts, rate limits, circuit
breakers and cost accounting, so the agents only say *what* they need.

Each task (``customer``, ``coach``, ``judge``) has an ordered chain of models,
possibly from different vendors. A call walks the chain:

* a model whose circuit breaker is open is skipped without a request;
* a retryable error (rate limit, timeout, 5xx) is retried with exponential
  backoff and jitter, then the next model is tried;
* a non-retryable error (bad request, auth) moves straight to the next model;
* if every model fails, ``AllModelsFailed`` is raised and the caller degrades.

Streaming falls back only *before* the first token. Once text has reached the
user, switching models would produce a reply stitched from two models, so a
mid-stream failure raises ``StreamInterrupted`` with the partial text instead.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from coach import telemetry as tm
from coach.llm.base import (
    AllModelsFailed,
    Completion,
    LLMError,
    Message,
    Provider,
    ProviderTimeout,
    Usage,
)
from coach.llm.breaker import CircuitBreaker, State
from coach.llm.pricing import cost_usd
from coach.llm.ratelimit import TokenBucket

logger = logging.getLogger("coach.router")


class StreamInterrupted(LLMError):
    retryable = False

    def __init__(self, partial_text: str, cause: Exception) -> None:
        super().__init__(f"stream interrupted after {len(partial_text)} chars: {cause}")
        self.partial_text = partial_text


@dataclass
class CallRecord:
    conversation_id: str
    task: str
    model: str
    attempt: int
    outcome: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class StreamResult:
    """Filled in while a stream runs; read it after the iterator is exhausted."""

    model: str = ""
    fallback_used: bool = False
    usage: Usage | None = None
    cost_usd: float = 0.0


class ModelRouter:
    def __init__(
        self,
        providers: dict[str, Provider],
        routes: dict[str, tuple[str, ...]],
        *,
        rate_limit_rpm: int = 600,
        timeout_s: float = 20.0,
        max_retries: int = 1,
        backoff_base_s: float = 0.2,
        breaker_failure_threshold: int = 5,
        breaker_reset_s: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        seed: int | None = None,
        on_call: Callable[[CallRecord], None] | None = None,
        prioritise_realtime: bool = True,
        realtime_tasks: frozenset[str] = frozenset({"customer"}),
    ) -> None:
        self.providers = providers
        self.routes = routes
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._sleep = sleep
        self._rng = random.Random(seed)
        self._on_call = on_call
        # One token bucket per vendor (a vendor's rate limit is per API key, not per model).
        # Live roleplay turns get priority in the queue when ``prioritise_realtime`` is on.
        self.realtime_tasks = realtime_tasks
        self.prioritise_realtime = prioritise_realtime
        self._buckets: dict[str, TokenBucket] = {}
        self._rate_limit_rpm = rate_limit_rpm
        self._breakers: dict[str, CircuitBreaker] = defaultdict(
            lambda: CircuitBreaker(breaker_failure_threshold, breaker_reset_s)
        )
        self._costs: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def vendor(model: str) -> str:
        return model.split("/", 1)[0]

    def _provider(self, model: str) -> Provider:
        provider = self.providers.get(self.vendor(model)) or self.providers.get("*")
        if provider is None:
            raise KeyError(f"no provider registered for model '{model}'")
        return provider

    def breaker(self, model: str) -> CircuitBreaker:
        return self._breakers[model]

    def conversation_cost(self, conversation_id: str) -> float:
        return self._costs.get(conversation_id, 0.0)

    def _record(self, record: CallRecord) -> None:
        tm.LLM_REQUESTS.labels(record.task, record.model, record.outcome).inc()
        if record.outcome == "ok":
            tm.LLM_LATENCY.labels(record.task, record.model).observe(record.latency_ms / 1000)
            self._costs[record.conversation_id] += record.cost_usd
        tm.BREAKER_OPEN.labels(record.model).set(
            1 if self.breaker(record.model).state is State.OPEN else 0
        )
        if self._on_call:
            self._on_call(record)

    async def _backoff(self, attempt: int) -> None:
        await self._sleep(self.backoff_base_s * (2**attempt) * (0.5 + self._rng.random()))

    def _bucket(self, vendor: str) -> TokenBucket:
        if vendor not in self._buckets:
            self._buckets[vendor] = TokenBucket(self._rate_limit_rpm)
        return self._buckets[vendor]

    async def _throttle(self, model: str, task: str) -> None:
        vendor = self.vendor(model)
        priority = 0 if self.prioritise_realtime and task in self.realtime_tasks else 1
        waited = await self._bucket(vendor).acquire(priority)
        if waited:
            tm.RATE_LIMIT_WAIT.labels(vendor).inc(waited)

    # ------------------------------------------------------------ complete
    async def complete(
        self,
        task: str,
        messages: list[Message],
        *,
        conversation_id: str = "",
        tools: list[dict] | None = None,
        json_mode: bool = False,
        temperature: float = 0.3,
    ) -> Completion:
        chain = self.routes[task]
        errors: list[str] = []
        for index, model in enumerate(chain):
            breaker = self.breaker(model)
            if not breaker.allow():
                errors.append(f"{model}: circuit open")
                self._record(CallRecord(conversation_id, task, model, 0, "skipped_open", 0.0))
                continue
            for attempt in range(self.max_retries + 1):
                await self._throttle(model, task)
                started = time.perf_counter()
                try:
                    with tm.span(
                        f"llm.{task}", model=model, attempt=attempt, conversation_id=conversation_id
                    ):
                        completion = await asyncio.wait_for(
                            self._provider(model).complete(
                                model,
                                messages,
                                tools=tools,
                                json_mode=json_mode,
                                temperature=temperature,
                                task=task,
                            ),
                            timeout=self.timeout_s,
                        )
                except TimeoutError:
                    error: LLMError = ProviderTimeout(f"no answer within {self.timeout_s}s")
                except LLMError as exc:
                    error = exc
                else:
                    latency_ms = (time.perf_counter() - started) * 1000
                    completion.latency_ms = latency_ms
                    if completion.cost_usd is None:
                        completion.cost_usd = cost_usd(model, completion.usage)
                    breaker.record_success()
                    if index > 0:
                        tm.LLM_FALLBACKS.labels(task).inc()
                    self._record(
                        CallRecord(
                            conversation_id,
                            task,
                            model,
                            attempt,
                            "ok",
                            latency_ms,
                            completion.usage.input_tokens,
                            completion.usage.output_tokens,
                            completion.cost_usd,
                        )
                    )
                    return completion

                latency_ms = (time.perf_counter() - started) * 1000
                breaker.record_failure()
                errors.append(f"{model}: {type(error).__name__}")
                self._record(
                    CallRecord(
                        conversation_id, task, model, attempt, type(error).__name__, latency_ms
                    )
                )
                tm.log(
                    logger,
                    logging.WARNING,
                    "llm call failed",
                    task=task,
                    model=model,
                    attempt=attempt,
                    error=str(error),
                    conversation_id=conversation_id,
                )
                if not error.retryable or breaker.state is not State.CLOSED:
                    break
                if attempt < self.max_retries:
                    await self._backoff(attempt)

        tm.LLM_EXHAUSTED.labels(task).inc()
        raise AllModelsFailed("; ".join(errors))

    # ------------------------------------------------------------ stream
    async def stream(
        self,
        task: str,
        messages: list[Message],
        result: StreamResult,
        *,
        conversation_id: str = "",
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        chain = self.routes[task]
        errors: list[str] = []
        for index, model in enumerate(chain):
            breaker = self.breaker(model)
            if not breaker.allow():
                errors.append(f"{model}: circuit open")
                self._record(CallRecord(conversation_id, task, model, 0, "skipped_open", 0.0))
                continue
            await self._throttle(model, task)
            started = time.perf_counter()
            parts: list[str] = []
            usage: Usage | None = None
            agen = self._provider(model).stream(model, messages, temperature=temperature, task=task)
            try:
                while True:
                    try:
                        chunk = await asyncio.wait_for(agen.__anext__(), timeout=self.timeout_s)
                    except StopAsyncIteration:
                        break
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if chunk.delta:
                        parts.append(chunk.delta)
                        yield chunk.delta
            except (TimeoutError, LLMError) as exc:
                error = exc if isinstance(exc, LLMError) else ProviderTimeout("stream stalled")
                breaker.record_failure()
                errors.append(f"{model}: {type(error).__name__}")
                latency_ms = (time.perf_counter() - started) * 1000
                self._record(
                    CallRecord(conversation_id, task, model, 0, type(error).__name__, latency_ms)
                )
                await agen.aclose()
                if parts:  # text already reached the user: do not stitch two models together
                    raise StreamInterrupted("".join(parts), error) from exc
                continue

            latency_ms = (time.perf_counter() - started) * 1000
            usage = usage or Usage()
            cost = cost_usd(model, usage)
            breaker.record_success()
            if index > 0:
                tm.LLM_FALLBACKS.labels(task).inc()
            result.model, result.fallback_used, result.usage, result.cost_usd = (
                model,
                index > 0,
                usage,
                cost,
            )
            self._record(
                CallRecord(
                    conversation_id,
                    task,
                    model,
                    0,
                    "ok",
                    latency_ms,
                    usage.input_tokens,
                    usage.output_tokens,
                    cost,
                )
            )
            return

        tm.LLM_EXHAUSTED.labels(task).inc()
        raise AllModelsFailed("; ".join(errors))
