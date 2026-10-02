"""Token prices for cost accounting.

Live calls take their cost from LiteLLM's own price map (``completion_cost``),
so no vendor prices are hard-coded here. The mock models get illustrative
prices so cost-per-conversation can be measured and alerted on offline.
Override or extend with the ``MODEL_PRICES`` environment variable, e.g.
``MODEL_PRICES="openai/<model-name>=0.15:0.60"`` (USD per 1M input:output tokens).
"""

from __future__ import annotations

import os

from coach.llm.base import Usage

# USD per 1M tokens (input, output). Illustrative values for the offline mock models.
_MOCK_PRICES: dict[str, tuple[float, float]] = {
    "mock/fast-a": (0.15, 0.60),
    "mock/fast-b": (0.25, 1.00),
    "mock/smart-a": (2.50, 10.00),
    "mock/smart-b": (3.00, 15.00),
}


def _env_prices() -> dict[str, tuple[float, float]]:
    prices: dict[str, tuple[float, float]] = {}
    for item in os.getenv("MODEL_PRICES", "").split(","):
        if "=" not in item:
            continue
        model, _, pair = item.partition("=")
        inp, _, out = pair.partition(":")
        prices[model.strip()] = (float(inp), float(out))
    return prices


PRICES: dict[str, tuple[float, float]] = {**_MOCK_PRICES, **_env_prices()}


def cost_usd(model: str, usage: Usage) -> float:
    inp, out = PRICES.get(model, (0.0, 0.0))
    return (usage.input_tokens * inp + usage.output_tokens * out) / 1_000_000
