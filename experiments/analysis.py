"""A/B test analysis with a decision rule written down before the data comes in.

For a primary metric (feedback score) and a guardrail metric (cost per
conversation) it reports:

* the difference in means with a percentile bootstrap 95% CI and a Welch t-test;
* a sample ratio mismatch (SRM) check, because a broken assignment invalidates
  everything else;
* the relative change in the guardrail with its own bootstrap CI;
* a decision: ``ship``, ``hold``, ``kill``, ``keep_running`` or ``invalid``.

The rule (defaults in ``DecisionRule``):

1. SRM p-value below 0.001 -> ``invalid``: fix assignment, do not read the result.
2. Primary CI entirely above 0 and the guardrail's upper CI bound within the
   allowed increase -> ``ship``.
3. Primary CI above 0 but the guardrail may break the budget -> ``hold``.
4. Primary CI entirely below the smallest effect worth having -> ``kill``.
5. Otherwise -> ``keep_running``.

Before rules 2-5, an arm below the planned sample size (80% power for the
minimum effect) always gets ``keep_running``: peeking at a small sample and
deciding is how noise gets shipped, or a good idea gets killed early.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DecisionRule:
    min_effect: float = 0.10  # smallest lift in mean feedback score worth shipping
    max_guardrail_increase: float = 0.10  # max relative cost increase allowed
    srm_alpha: float = 0.001
    expected_split: float = 0.5  # share of units expected in treatment
    n_boot: int = 4000
    seed: int = 20261003


@dataclass
class ExperimentReport:
    control: str
    treatment: str
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    diff: float
    diff_ci: tuple[float, float]
    p_value: float
    srm_p_value: float
    guardrail_rel_change: float
    guardrail_rel_ci: tuple[float, float]
    decision: str
    reason: str
    n_per_arm_for_min_effect: int

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    diffs = b[ib].mean(axis=1) - a[ia].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_rel(
    a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    rel = b[ib].mean(axis=1) / a[ia].mean(axis=1) - 1
    lo, hi = np.percentile(rel, [2.5, 97.5])
    return float(lo), float(hi)


def srm_p_value(n_control: int, n_treatment: int, expected_split: float = 0.5) -> float:
    total = n_control + n_treatment
    expected = [total * (1 - expected_split), total * expected_split]
    return float(stats.chisquare([n_control, n_treatment], f_exp=expected).pvalue)


def required_n_per_arm(
    sd: float, min_effect: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return math.ceil(2 * (z * sd / min_effect) ** 2)


def analyse(
    control_metric: np.ndarray,
    treatment_metric: np.ndarray,
    control_guardrail: np.ndarray,
    treatment_guardrail: np.ndarray,
    *,
    control: str = "control",
    treatment: str = "treatment",
    rule: DecisionRule | None = None,
) -> ExperimentReport:
    rule = rule or DecisionRule()
    rng = np.random.default_rng(rule.seed)
    a, b = np.asarray(control_metric, float), np.asarray(treatment_metric, float)
    ga, gb = np.asarray(control_guardrail, float), np.asarray(treatment_guardrail, float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("need at least two units per arm")

    diff = float(b.mean() - a.mean())
    diff_ci = _bootstrap_diff(a, b, rule.n_boot, rng)
    p_value = float(stats.ttest_ind(b, a, equal_var=False).pvalue)
    srm = srm_p_value(len(a), len(b), rule.expected_split)
    g_rel = float(gb.mean() / ga.mean() - 1) if ga.mean() > 0 else 0.0
    g_ci = _bootstrap_rel(ga, gb, rule.n_boot, rng) if ga.mean() > 0 else (0.0, 0.0)
    pooled_sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
    n_needed = required_n_per_arm(pooled_sd, rule.min_effect) if pooled_sd > 0 else 2

    if srm < rule.srm_alpha:
        decision, reason = (
            "invalid",
            f"sample ratio mismatch (p={srm:.2g}); fix assignment before reading results",
        )
    elif min(len(a), len(b)) < n_needed:
        # Fixed-horizon test: no decision before the planned sample size, in either direction.
        decision, reason = (
            "keep_running",
            f"below the planned {n_needed} units per arm; do not decide yet",
        )
    elif diff_ci[0] > 0 and g_ci[1] <= rule.max_guardrail_increase:
        decision, reason = "ship", "primary metric improved and cost stays within the guardrail"
    elif diff_ci[0] > 0:
        decision, reason = "hold", f"primary metric improved but cost may rise up to {g_ci[1]:+.0%}"
    elif diff_ci[1] < rule.min_effect:
        decision, reason = (
            "kill",
            f"the effect is below the {rule.min_effect} minimum worth shipping, even at the CI's upper end",
        )
    else:
        decision, reason = (
            "keep_running",
            f"inconclusive; about {n_needed} units per arm detect the minimum effect",
        )

    return ExperimentReport(
        control,
        treatment,
        len(a),
        len(b),
        float(a.mean()),
        float(b.mean()),
        diff,
        diff_ci,
        p_value,
        srm,
        g_rel,
        g_ci,
        decision,
        reason,
        n_needed,
    )


def analyse_rows(
    rows: list[dict],
    *,
    control: str,
    treatment: str,
    metric: str = "overall",
    guardrail: str = "cost_usd",
    rule: DecisionRule | None = None,
) -> ExperimentReport:
    """Analyse rows as returned by ``SessionStore.experiment_rows()``."""
    a = [r for r in rows if r["variant"] == control]
    b = [r for r in rows if r["variant"] == treatment]
    return analyse(
        np.array([r[metric] for r in a]),
        np.array([r[metric] for r in b]),
        np.array([r[guardrail] for r in a]),
        np.array([r[guardrail] for r in b]),
        control=control,
        treatment=treatment,
        rule=rule,
    )
