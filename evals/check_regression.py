"""Fail CI when a judge or pipeline change makes the offline eval worse.

    python -m evals.check_regression reports/eval_keyword-baseline.json evals/baseline_keyword.json

Compares the overall quadratic weighted kappa and exact agreement against a
committed baseline and exits non-zero if either drops by more than the
tolerance. Improvements pass; update the baseline in the same pull request.
"""

from __future__ import annotations

import json
import sys

TOLERANCE = {"qwk": 0.03, "exact": 0.03}


def main(report_path: str, baseline_path: str) -> int:
    report = json.load(open(report_path))["overall"]
    baseline = json.load(open(baseline_path))["overall"]
    failed = False
    for metric, tol in TOLERANCE.items():
        delta = report[metric] - baseline[metric]
        status = "FAIL" if delta < -tol else "ok"
        failed |= status == "FAIL"
        print(
            f"{status:4} {metric}: {report[metric]:.3f} (baseline {baseline[metric]:.3f}, {delta:+.3f})"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
