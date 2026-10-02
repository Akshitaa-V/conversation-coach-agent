"""Live provider: any model LiteLLM supports (OpenAI, Anthropic, Gemini, Vertex AI, ...).

Model names use LiteLLM's ``provider/model`` form, e.g. ``openai/<model-name>``,
``anthropic/<model-name>`` or ``vertex_ai/<model-name>``.
API keys come from the usual environment variables (``OPENAI_API_KEY`` and so
on); on Cloud Run they are mounted from Secret Manager.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from coach.llm.base import (
    BadRequest,
    Completion,
    Message,
    ProviderError,
    ProviderTimeout,
    RateLimited,
    StreamChunk,
    ToolCall,
    Usage,
)


class LiteLLMProvider:
    def __init__(self) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("live mode needs the 'live' extra: pip install '.[live]'") from exc
        import litellm

        self._litellm = litellm

    def _map_error(self, exc: Exception) -> Exception:
        name = type(exc).__name__
        if "RateLimit" in name:
            return RateLimited(str(exc))
        if "Timeout" in name:
            return ProviderTimeout(str(exc))
        if name in {
            "BadRequestError",
            "AuthenticationError",
            "NotFoundError",
            "ContextWindowExceededError",
        }:
            return BadRequest(str(exc))
        return ProviderError(f"{name}: {exc}")

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
        kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
        if json_mode and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = await self._litellm.acompletion(**kwargs)
        except Exception as exc:  # noqa: BLE001 - vendor SDKs raise many types
            raise self._map_error(exc) from exc

        message = response.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (getattr(message, "tool_calls", None) or [])
        ]
        usage = Usage(
            getattr(response.usage, "prompt_tokens", 0) or 0,
            getattr(response.usage, "completion_tokens", 0) or 0,
        )
        try:
            cost = float(self._litellm.completion_cost(completion_response=response))
        except Exception:  # noqa: BLE001 - unknown price for this model
            cost = None
        return Completion(
            text=message.content or "",
            model=model,
            usage=usage,
            tool_calls=tool_calls,
            cost_usd=cost,
        )

    async def stream(
        self,
        model: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        task: str = "",
    ) -> AsyncIterator[StreamChunk]:
        try:
            response = await self._litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            usage = None
            async for chunk in response:
                if getattr(chunk, "usage", None):
                    usage = Usage(
                        chunk.usage.prompt_tokens or 0, chunk.usage.completion_tokens or 0
                    )
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield StreamChunk(delta=chunk.choices[0].delta.content)
            yield StreamChunk(usage=usage or Usage())
        except Exception as exc:  # noqa: BLE001
            raise self._map_error(exc) from exc
