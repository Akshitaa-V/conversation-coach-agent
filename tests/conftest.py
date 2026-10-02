from __future__ import annotations

import pytest

from coach.config import Settings
from coach.llm.mock import MockProvider
from coach.llm.router import ModelRouter
from coach.llm.scripted import make_responder
from coach.store import SessionStore


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture
def make_router():
    def factory(
        routes: dict[str, tuple[str, ...]] | None = None,
        *,
        failure_rate: dict[str, float] | None = None,
        midstream_failure_rate: dict[str, float] | None = None,
        hang_models: set[str] | None = None,
        bad_json_rate: float = 0.0,
        **kwargs,
    ) -> tuple[ModelRouter, MockProvider]:
        provider = MockProvider(
            make_responder(bad_json_rate),
            latency_ms=1,
            token_delay_ms=0,
            failure_rate=failure_rate,
            midstream_failure_rate=midstream_failure_rate,
            hang_models=hang_models,
        )
        router = ModelRouter(
            {"mock": provider},
            routes
            or {
                "customer": ("mock/fast-a", "mock/fast-b"),
                "coach": ("mock/smart-a", "mock/smart-b"),
                "judge": ("mock/smart-b",),
                "judge_pairwise": ("mock/smart-b",),
            },
            sleep=_no_sleep,
            seed=1,
            **{"rate_limit_rpm": 60_000, **kwargs},
        )
        return router, provider

    return factory


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        db_path=str(tmp_path / "test.db"),
        mock_latency_ms=1,
        mock_token_delay_ms=0,
        backoff_base_s=0.0,
        rate_limit_rpm=60_000,
        online_eval_sample_rate=1.0,
    )


@pytest.fixture
def store(settings) -> SessionStore:
    return SessionStore(settings.db_path)
