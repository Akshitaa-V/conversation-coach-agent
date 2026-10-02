"""Logs, metrics and traces.

- Logs are single-line JSON on stdout, which Cloud Logging parses into structured
  entries (``severity`` and ``message`` are the field names it expects).
- Metrics are Prometheus counters and histograms served on ``/metrics``.
- Traces use OpenTelemetry when the ``otel`` extra is installed and
  ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set; otherwise ``span()`` is a no-op.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------- metrics
LLM_REQUESTS = Counter(
    "coach_llm_requests_total", "LLM calls by outcome", ["task", "model", "outcome"]
)
LLM_LATENCY = Histogram(
    "coach_llm_latency_seconds",
    "LLM call latency",
    ["task", "model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
)
LLM_FALLBACKS = Counter(
    "coach_llm_fallbacks_total", "Requests served by a non-primary model", ["task"]
)
LLM_EXHAUSTED = Counter("coach_llm_exhausted_total", "Requests where every model failed", ["task"])
BREAKER_OPEN = Gauge("coach_breaker_open", "1 while a model's circuit breaker is open", ["model"])
RATE_LIMIT_WAIT = Counter(
    "coach_rate_limit_wait_seconds_total", "Time spent throttled", ["provider"]
)
TTFT = Histogram(
    "coach_ws_time_to_first_token_seconds",
    "Time from trainee message to first streamed customer token",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 2, 4),
)
DEGRADED_REPLIES = Counter(
    "coach_degraded_replies_total", "Customer replies served from the canned fallback"
)
CONVERSATION_COST = Histogram(
    "coach_conversation_cost_usd",
    "LLM cost per finished conversation",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
)
FEEDBACK_REPAIRS = Counter("coach_feedback_repairs_total", "Structured-output repairs", ["reason"])
ONLINE_JUDGE_GAP = Histogram(
    "coach_online_judge_gap",
    "Absolute gap between coach overall score and sampled judge overall score",
    buckets=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
)


# ---------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(level: str | None = None) -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(level or os.getenv("LOG_LEVEL", "INFO"))


def log(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    logger.log(level, message, extra={"fields": fields})


# ---------------------------------------------------------------- tracing
_tracer = None


def setup_tracing(service_name: str = "conversation-coach-agent") -> bool:
    """Enable OpenTelemetry export if the extra is installed and an endpoint is configured."""
    global _tracer
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    return True


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(
                    key, value if isinstance(value, (str, int, float, bool)) else str(value)
                )
        yield current
