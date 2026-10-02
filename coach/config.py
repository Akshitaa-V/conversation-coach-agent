"""Runtime settings, read once from environment variables.

Everything has a safe default so the service starts offline in ``mock`` mode,
which is what the tests, the load test and CI use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in os.getenv(name, default).split(",") if part.strip())


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # "mock" runs fully offline with deterministic providers; "live" routes through LiteLLM.
    provider_mode: str = "mock"

    # Ordered fallback chains per task. The first model is primary.
    customer_models: tuple[str, ...] = ("mock/fast-a", "mock/fast-b")
    coach_models: tuple[str, ...] = ("mock/smart-a", "mock/smart-b")
    judge_models: tuple[str, ...] = ("mock/smart-b",)

    request_timeout_s: float = 20.0
    max_retries: int = 1
    backoff_base_s: float = 0.2
    breaker_failure_threshold: int = 5
    breaker_reset_s: float = 30.0

    # Per-provider token bucket (requests per minute).
    rate_limit_rpm: int = 600
    # Live roleplay turns jump the rate-limit queue ahead of feedback and judge calls.
    prioritise_realtime: bool = True

    memory_token_budget: int = 1200
    max_turns: int = 12

    db_path: str = "coach.db"
    online_eval_sample_rate: float = 0.2
    experiment_name: str = "customer_prompt_v2"

    # Mock provider behaviour (ignored in live mode).
    mock_latency_ms: int = 40
    mock_token_delay_ms: int = 5
    mock_failure_rate: dict[str, float] = field(default_factory=dict)
    mock_bad_json_rate: float = 0.0

    @classmethod
    def from_env(cls) -> Settings:
        failure_rates: dict[str, float] = {}
        for item in _csv("MOCK_FAILURE_RATES", ""):
            model, _, rate = item.partition("=")
            failure_rates[model.strip()] = float(rate)
        return cls(
            provider_mode=os.getenv("PROVIDER_MODE", "mock"),
            customer_models=_csv("CUSTOMER_MODELS", "mock/fast-a,mock/fast-b"),
            coach_models=_csv("COACH_MODELS", "mock/smart-a,mock/smart-b"),
            judge_models=_csv("JUDGE_MODELS", "mock/smart-b"),
            request_timeout_s=_float("REQUEST_TIMEOUT_S", 20.0),
            max_retries=_int("MAX_RETRIES", 1),
            backoff_base_s=_float("BACKOFF_BASE_S", 0.2),
            breaker_failure_threshold=_int("BREAKER_FAILURE_THRESHOLD", 5),
            breaker_reset_s=_float("BREAKER_RESET_S", 30.0),
            rate_limit_rpm=_int("RATE_LIMIT_RPM", 600),
            prioritise_realtime=os.getenv("PRIORITISE_REALTIME", "true").lower() == "true",
            memory_token_budget=_int("MEMORY_TOKEN_BUDGET", 1200),
            max_turns=_int("MAX_TURNS", 12),
            db_path=os.getenv("DB_PATH", "coach.db"),
            online_eval_sample_rate=_float("ONLINE_EVAL_SAMPLE_RATE", 0.2),
            experiment_name=os.getenv("EXPERIMENT_NAME", "customer_prompt_v2"),
            mock_latency_ms=_int("MOCK_LATENCY_MS", 40),
            mock_token_delay_ms=_int("MOCK_TOKEN_DELAY_MS", 5),
            mock_failure_rate=failure_rates,
            mock_bad_json_rate=_float("MOCK_BAD_JSON_RATE", 0.0),
        )
