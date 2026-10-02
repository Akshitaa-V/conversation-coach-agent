"""Deterministic offline provider with fault injection.

Used by the tests, the load test and CI. It behaves like a real vendor where it
matters for the surrounding system: latency, token-by-token streaming, usage
counts, rate-limit errors, timeouts, 5xx errors, failures in the middle of a
stream and malformed JSON. The *content* comes from scripted responders in
``coach.llm.scripted``; it is not meant to be a good model.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable

from coach.llm.base import (
    Completion,
    Message,
    ProviderError,
    ProviderTimeout,
    RateLimited,
    StreamChunk,
    Usage,
    estimate_tokens,
    message_tokens,
)

Responder = Callable[[str, list[Message], list[dict] | None, bool], Completion]


class MockProvider:
    def __init__(
        self,
        responder: Responder,
        *,
        latency_ms: int = 40,
        token_delay_ms: int = 5,
        failure_rate: dict[str, float] | None = None,
        midstream_failure_rate: dict[str, float] | None = None,
        hang_models: set[str] | None = None,
        seed: int = 7,
    ) -> None:
        self.responder = responder
        self.latency_ms = latency_ms
        self.token_delay_ms = token_delay_ms
        self.failure_rate = failure_rate or {}
        self.midstream_failure_rate = midstream_failure_rate or {}
        self.hang_models = hang_models or set()
        self._rng = random.Random(seed)
        self.calls: list[str] = []

    async def _maybe_fail(self, model: str) -> None:
        self.calls.append(model)
        if model in self.hang_models:
            await asyncio.sleep(3600)  # the router's timeout has to cut this off
        await asyncio.sleep(self.latency_ms / 1000 * (0.5 + self._rng.random()))
        if self._rng.random() < self.failure_rate.get(model, 0.0):
            kind = self._rng.choice((RateLimited, ProviderTimeout, ProviderError))
            raise kind(f"injected {kind.__name__} from {model}")

    async def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        json_mode: bool = False,
        temperature: float = 0.3,
        task: str = "",
    ) -> Completion:
        await self._maybe_fail(model)
        completion = self.responder(model, messages, tools, json_mode)
        completion.model = model
        completion.usage = Usage(message_tokens(messages), estimate_tokens(completion.text or "x"))
        return completion

    async def stream(
        self,
        model: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        task: str = "",
    ) -> AsyncIterator[StreamChunk]:
        await self._maybe_fail(model)
        text = self.responder(model, messages, None, False).text
        words = text.split(" ")
        fail_at = None
        if self._rng.random() < self.midstream_failure_rate.get(model, 0.0):
            fail_at = max(1, len(words) // 2)
        for i, word in enumerate(words):
            if fail_at is not None and i == fail_at:
                raise ProviderError(f"injected stream drop from {model}")
            await asyncio.sleep(self.token_delay_ms / 1000)
            yield StreamChunk(delta=word if i == 0 else " " + word)
        yield StreamChunk(usage=Usage(message_tokens(messages), estimate_tokens(text)))
