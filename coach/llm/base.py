"""Provider-neutral types for LLM calls.

Messages use the OpenAI chat format (plain dicts) because LiteLLM accepts it for
every provider, which keeps the agents independent of the vendor behind them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


class LLMError(Exception):
    """Base class. ``retryable`` decides whether the router retries the same model."""

    retryable = True


class RateLimited(LLMError):
    pass


class ProviderTimeout(LLMError):
    pass


class ProviderError(LLMError):
    pass


class BadRequest(LLMError):
    retryable = False


class AllModelsFailed(LLMError):
    retryable = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string, as returned by the provider


@dataclass
class Completion:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    cost_usd: float | None = None  # filled by the provider when it knows, else by the router
    latency_ms: float = 0.0


@dataclass
class StreamChunk:
    delta: str = ""
    usage: Usage | None = None  # only on the final chunk


class Provider(Protocol):
    async def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        json_mode: bool = False,
        temperature: float = 0.3,
        task: str = "",
    ) -> Completion: ...

    def stream(
        self, model: str, messages: list[Message], *, temperature: float = 0.7, task: str = ""
    ) -> AsyncIterator[StreamChunk]: ...


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token) for budgets and mock usage."""
    return max(1, len(text) // 4)


def message_tokens(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        content = m.get("content") or ""
        total += estimate_tokens(content if isinstance(content, str) else str(content)) + 4
    return total
