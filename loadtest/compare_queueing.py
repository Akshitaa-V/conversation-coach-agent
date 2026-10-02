"""Experiment: does realtime-first rate-limit queueing cut time to first token?

Hypothesis: under rate-limit pressure, live roleplay turns wait behind feedback
and judge calls in a shared FIFO queue, which shows up as a long tail in time to
first token. Giving live turns priority should cut the p95 at the cost of slower
feedback. Each arm runs three times with no injected failures, so the only
difference is the queueing policy.

    python -m loadtest.compare_queueing     # writes reports/queueing_experiment.{json,md}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

from loadtest.run_load import run_local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    import logging

    from coach.telemetry import setup_logging

    setup_logging("WARNING")
    logging.getLogger("coach").setLevel(logging.ERROR)

    results: dict[str, list[dict]] = {"fifo": [], "realtime_first": []}
    for arm, no_priority in (("fifo", True), ("realtime_first", False)):
        for i in range(args.repeats):
            db = Path(f"/tmp/queueing_{arm}_{i}.db")
            db.unlink(missing_ok=True)
            ns = SimpleNamespace(
                db=str(db),
                failure_rate=0.0,
                bad_json_rate=0.0,
                latency_ms=60,
                rate_limit_rpm=6000,
                no_priority=no_priority,
                conversations=200,
                concurrency=40,
                turns=4,
                seed=5 + i,
            )
            report = asyncio.run(run_local(ns))
            results[arm].append(
                {
                    "ttft_p95_ms": report["ttft_ms"]["p95"],
                    "ttft_p50_ms": report["ttft_ms"]["p50"],
                    "feedback_p95_ms": report["feedback_ms"]["p95"],
                    "errors": report["user_visible_errors"],
                }
            )

    def mean(arm: str, key: str) -> float:
        return statistics.fmean(r[key] for r in results[arm])

    lines = [
        "# Rate-limit queueing experiment",
        "",
        "200 conversations x 4 turns at concurrency 40, 6000 requests/min per vendor, no injected failures, "
        f"{args.repeats} runs per arm (mean shown).",
        "",
        "| Policy | TTFT p50 | TTFT p95 | Feedback p95 | Errors |",
        "|---|---|---|---|---|",
    ]
    for arm in results:
        lines.append(
            f"| {arm} | {mean(arm, 'ttft_p50_ms'):.0f} ms | {mean(arm, 'ttft_p95_ms'):.0f} ms | "
            f"{mean(arm, 'feedback_p95_ms'):.0f} ms | {sum(r['errors'] for r in results[arm])} |"
        )
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "queueing_experiment.json").write_text(json.dumps(results, indent=2))
    (out / "queueing_experiment.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
