"""A/B assignment for prompt and model variants.

Assignment is a hash of (experiment, trainee), so a trainee keeps the same
variant across sessions and devices, and no assignment table is needed. The
variant is stored on every session, and ``experiments/analysis.py`` reads it back.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    customer_instructions: str
    temperature: float


VARIANTS: dict[str, Variant] = {
    "control": Variant(
        "control",
        "Reply in one or two short sentences.",
        0.7,
    ),
    "realism_v2": Variant(
        "realism_v2",
        "Reply in one to three sentences. Sound like a real, busy person: hesitate, hedge, "
        "and refer back to what the trainee said earlier when it matters to you.",
        0.8,
    ),
}


def assign(experiment: str, unit_id: str, split: dict[str, float] | None = None) -> str:
    """Deterministically map a unit (trainee) to a variant according to ``split``."""
    split = split or {"control": 0.5, "realism_v2": 0.5}
    if abs(sum(split.values()) - 1.0) > 1e-9:
        raise ValueError("split must sum to 1")
    digest = hashlib.sha256(f"{experiment}:{unit_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    cumulative = 0.0
    for name, share in split.items():
        cumulative += share
        if bucket < cumulative:
            return name
    return next(reversed(split))
