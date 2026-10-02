from __future__ import annotations

import asyncio

import pytest

from coach.llm.base import AllModelsFailed, Completion, ProviderError
from coach.llm.breaker import CircuitBreaker, State
from coach.llm.ratelimit import TokenBucket
from coach.llm.router import StreamInterrupted, StreamResult

CUSTOMER_MSGS = [
    {
        "role": "system",
        "content": "[task:customer]\n(scenario: billing_complaint) (variant: control)",
    },
    {"role": "user", "content": "I'm sorry about the double charge."},
]


async def test_primary_success_records_cost(make_router):
    router, provider = make_router()
    completion = await router.complete("customer", CUSTOMER_MSGS, conversation_id="c1")
    assert completion.model == "mock/fast-a"
    assert completion.cost_usd > 0
    assert router.conversation_cost("c1") == pytest.approx(completion.cost_usd)
    assert provider.calls == ["mock/fast-a"]


async def test_falls_back_to_next_model_when_primary_fails(make_router):
    router, provider = make_router(failure_rate={"mock/fast-a": 1.0}, max_retries=1)
    completion = await router.complete("customer", CUSTOMER_MSGS)
    assert completion.model == "mock/fast-b"
    assert provider.calls == [
        "mock/fast-a",
        "mock/fast-a",
        "mock/fast-b",
    ]  # one retry, then fallback


async def test_all_models_failing_raises(make_router):
    router, _ = make_router(failure_rate={"mock/fast-a": 1.0, "mock/fast-b": 1.0})
    with pytest.raises(AllModelsFailed):
        await router.complete("customer", CUSTOMER_MSGS)


async def test_timeout_moves_on_to_fallback(make_router):
    router, _ = make_router(hang_models={"mock/fast-a"}, timeout_s=0.05, max_retries=0)
    completion = await router.complete("customer", CUSTOMER_MSGS)
    assert completion.model == "mock/fast-b"


async def test_non_retryable_error_is_not_retried(make_router):
    router, provider = make_router(max_retries=3)

    class BadRequestProvider:
        async def complete(self, model, messages, **kwargs):
            provider.calls.append(model)
            from coach.llm.base import BadRequest

            raise BadRequest("invalid request")

    router.providers["bad"] = BadRequestProvider()
    router.routes["customer"] = ("bad/model", "mock/fast-b")
    completion = await router.complete("customer", CUSTOMER_MSGS)
    assert completion.model == "mock/fast-b"
    assert provider.calls.count("bad/model") == 1


async def test_open_breaker_skips_model_without_a_request(make_router):
    router, provider = make_router(
        failure_rate={"mock/fast-a": 1.0}, max_retries=0, breaker_failure_threshold=2
    )
    for _ in range(2):
        await router.complete("customer", CUSTOMER_MSGS)
    assert router.breaker("mock/fast-a").state is State.OPEN
    provider.calls.clear()
    await router.complete("customer", CUSTOMER_MSGS)
    assert provider.calls == ["mock/fast-b"]


def test_breaker_half_open_lets_one_probe_through():
    now = [0.0]
    breaker = CircuitBreaker(failure_threshold=1, reset_after_s=10, clock=lambda: now[0])
    breaker.record_failure()
    assert not breaker.allow()
    now[0] = 11
    assert breaker.allow()
    assert not breaker.allow()  # only one probe
    breaker.record_success()
    assert breaker.state is State.CLOSED


async def test_token_bucket_throttles_bursts():
    bucket = TokenBucket(rate_per_minute=600, burst=2)  # 10 per second
    waits = [await bucket.acquire() for _ in range(4)]
    assert waits[0] == 0 and waits[1] == 0
    assert sum(waits) == pytest.approx(0.2, abs=0.05)


async def test_concurrent_requests_respect_rate_limit():
    bucket = TokenBucket(rate_per_minute=1200, burst=1)  # 20 per second
    loop = asyncio.get_running_loop()
    start = loop.time()
    await asyncio.gather(*(bucket.acquire() for _ in range(5)))
    assert loop.time() - start >= 0.18


async def test_stream_falls_back_before_first_token(make_router):
    router, _ = make_router(failure_rate={"mock/fast-a": 1.0}, max_retries=0)
    result = StreamResult()
    text = "".join([d async for d in router.stream("customer", CUSTOMER_MSGS, result)])
    assert text
    assert result.model == "mock/fast-b" and result.fallback_used
    assert result.cost_usd > 0


async def test_stream_failure_after_first_token_is_not_stitched(make_router):
    router, provider = make_router(midstream_failure_rate={"mock/fast-a": 1.0})
    result = StreamResult()
    received = []
    with pytest.raises(StreamInterrupted) as info:
        async for delta in router.stream("customer", CUSTOMER_MSGS, result):
            received.append(delta)
    assert info.value.partial_text == "".join(received)
    assert "mock/fast-b" not in provider.calls


async def test_provider_cost_is_kept_when_reported(make_router):
    router, _ = make_router()

    class PricedProvider:
        async def complete(self, model, messages, **kwargs):
            return Completion(text="hi", model=model, cost_usd=0.123)

    router.providers["priced"] = PricedProvider()
    router.routes["customer"] = ("priced/model",)
    await router.complete("customer", CUSTOMER_MSGS, conversation_id="c9")
    assert router.conversation_cost("c9") == pytest.approx(0.123)


async def test_records_every_attempt(make_router):
    records = []
    router, _ = make_router(
        failure_rate={"mock/fast-a": 1.0}, max_retries=1, on_call=records.append
    )
    await router.complete("customer", CUSTOMER_MSGS, conversation_id="c2")
    outcomes = [(r.model, r.outcome) for r in records]
    assert outcomes[-1] == ("mock/fast-b", "ok")
    assert len([o for o in outcomes if o[0] == "mock/fast-a"]) == 2


def test_error_classes():
    assert ProviderError.retryable is True


async def test_realtime_requests_jump_the_rate_limit_queue():
    bucket = TokenBucket(rate_per_minute=600, burst=1)  # one token, then 10 per second
    await bucket.acquire()
    order: list[str] = []

    async def take(name: str, priority: int) -> None:
        await bucket.acquire(priority)
        order.append(name)

    batch = [asyncio.create_task(take(f"batch{i}", 1)) for i in range(3)]
    await asyncio.sleep(0)  # batch requests are queued first
    live = asyncio.create_task(take("live", 0))
    await asyncio.gather(*batch, live)
    assert order[0] == "live"
    assert order[1:] == ["batch0", "batch1", "batch2"]  # FIFO within a priority


async def test_router_gives_customer_turns_priority(make_router):
    router, _ = make_router(rate_limit_rpm=600)
    assert router.prioritise_realtime and "customer" in router.realtime_tasks
