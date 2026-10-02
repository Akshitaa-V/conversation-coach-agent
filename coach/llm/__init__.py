"""LLM access layer: provider-neutral types, routing, fallbacks and cost accounting."""

from coach.llm.base import AllModelsFailed, Completion, LLMError, Message
from coach.llm.router import CallRecord, ModelRouter, StreamInterrupted, StreamResult

__all__ = [
    "AllModelsFailed",
    "CallRecord",
    "Completion",
    "LLMError",
    "Message",
    "ModelRouter",
    "StreamInterrupted",
    "StreamResult",
]
