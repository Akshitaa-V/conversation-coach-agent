"""Check the A/B pipeline against cases where the right answer is known.

An analysis is only trustworthy if it (a) finds an effect that is really there,
(b) does not find one that is not, and (c) refuses to read a broken experiment.
This script plants each situation in simulated session data and checks the
decision, then repeats A/A and planted-effect runs to measure the false-ship
rate and CI coverage.

    python -m experiments.simulate_ab            # writes reports/ab_simulation.{json,md}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analysis import DecisionRule, analyse

SCORE_SD = 0.7
BASE_SCORE = 3.1
BASE_COST = 0.004  # USD per conversation


def simulate_arm(
    rng: np.random.Generator, n: int, lift: float, cost_lift: float
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.clip(rng.normal(BASE_SCORE + lift, SCORE_SD, n), 1, 5)
    costs = rng.lognormal(np.log(BASE_COST * (1 + cost_lift)), 0.25, n)
    return scores, costs


CASES = {
    # name: (n_control, n_treatment, score lift, cost lift, expected decision)
    "planted_effect_cheap": (1500, 1500, 0.25, 0.04, "ship"),
    "planted_effect_costly": (1500, 1500, 0.25, 0.30, "hold"),
    "no_effect_costly": (2000, 2000, 0.0, 0.25, "kill"),
    "too_few_sessions": (60, 60, 0.12, 0.0, "keep_running"),
    "broken_assignment": (2150, 1850, 0.25, 0.04, "invalid"),
}


def run_cases(seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for name, (nc, nt, lift, cost_lift, expected) in CASES.items():
        ca, ga = simulate_arm(rng, nc, 0.0, 0.0)
        cb, gb = simulate_arm(rng, nt, lift, cost_lift)
        report = analyse(ca, cb, ga, gb, control="control", treatment="treatment")
        out.append(
            {
                "case": name,
                "planted_lift": lift,
                "planted_cost_lift": cost_lift,
                "expected": expected,
                **report.to_dict(),
                "correct": report.decision == expected,
            }
        )
    return out


def calibration(seed: int, runs: int, n: int) -> dict:
    """A/A runs measure false ships; planted runs measure CI coverage and detection rate."""
    rng = np.random.default_rng(seed)
    rule = DecisionRule(n_boot=1000)
    false_ship = covered = detected = 0
    lift = 0.15
    for _ in range(runs):
        ca, ga = simulate_arm(rng, n, 0.0, 0.0)
        cb, gb = simulate_arm(rng, n, 0.0, 0.0)
        if analyse(ca, cb, ga, gb, rule=rule).decision == "ship":
            false_ship += 1
        ca, ga = simulate_arm(rng, n, 0.0, 0.0)
        cb, gb = simulate_arm(rng, n, lift, 0.0)
        report = analyse(ca, cb, ga, gb, rule=rule)
        true_diff = float(
            np.clip(
                np.random.default_rng(0).normal(BASE_SCORE + lift, SCORE_SD, 400_000), 1, 5
            ).mean()
            - np.clip(np.random.default_rng(1).normal(BASE_SCORE, SCORE_SD, 400_000), 1, 5).mean()
        )
        covered += report.diff_ci[0] <= true_diff <= report.diff_ci[1]
        detected += report.diff_ci[0] > 0
    return {
        "runs": runs,
        "n_per_arm": n,
        "aa_false_ship_rate": false_ship / runs,
        "planted_lift": lift,
        "ci_coverage": covered / runs,
        "detection_rate": detected / runs,
    }


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.2g}"


def to_markdown(cases: list[dict], calib: dict) -> str:
    lines = [
        "# A/B pipeline check on planted effects",
        "",
        "Simulated session data with known ground truth. Score = mean feedback score (1-5); "
        "guardrail = LLM cost per conversation.",
        "",
        "| Case | Planted lift | Planted cost change | Diff (95% CI) | Cost change (95% CI) | SRM p | Decision | Expected |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        lines.append(
            f"| {c['case']} | {c['planted_lift']:+.2f} | {c['planted_cost_lift']:+.0%} | "
            f"{c['diff']:+.3f} ({c['diff_ci'][0]:+.3f}, {c['diff_ci'][1]:+.3f}) | "
            f"{c['guardrail_rel_change']:+.1%} ({c['guardrail_rel_ci'][0]:+.1%}, {c['guardrail_rel_ci'][1]:+.1%}) | "
            f"{_p(c['srm_p_value'])} | **{c['decision']}** | {c['expected']} |"
        )
    lines += [
        "",
        f"Correct decisions: {sum(c['correct'] for c in cases)}/{len(cases)}",
        "",
        "## Calibration",
        "",
        f"- A/A runs ({calib['runs']} x {calib['n_per_arm']} per arm): false ship rate "
        f"**{calib['aa_false_ship_rate']:.1%}** (a two-sided 95% CI should ship by chance in about 2.5% of runs)",
        f"- Planted lift of {calib['planted_lift']} : 95% CI covered the true difference in "
        f"**{calib['ci_coverage']:.1%}** of runs; effect detected in {calib['detection_rate']:.1%}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--runs", type=int, default=400)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    cases = run_cases(args.seed)
    calib = calibration(args.seed + 1, args.runs, args.n)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "ab_simulation.json").write_text(
        json.dumps({"cases": cases, "calibration": calib}, indent=2)
    )
    (out / "ab_simulation.md").write_text(to_markdown(cases, calib))
    print(to_markdown(cases, calib))


if __name__ == "__main__":
    main()
