"""Offline eval: how well does a judge agree with the labels, and where does it break?

    python -m evals.run_eval                    # keyword baseline, offline
    PROVIDER_MODE=live JUDGE_MODELS=openai/<model-name> python -m evals.run_eval --judge llm

Reports, per rubric dimension and per phrasing type (explicit / implicit / decoy):
exact agreement, within-one agreement, quadratic weighted kappa and bias. Two
probes look for known judge failure modes:

* position bias: compare pairs of transcripts as (A, B) and again as (B, A);
  a consistent judge flips its verdict;
* verbosity bias: pad every trainee turn with polite filler that adds no skill;
  the scores should not move.

Results go to ``reports/eval_<judge>.json`` and ``.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from coach.app import build_router
from coach.config import Settings
from coach.judge import KeywordJudge, LLMJudge
from coach.schemas import DIMENSIONS
from evals.metrics import agreement, spearman

DATA = Path(__file__).parent / "data" / "transcripts.jsonl"
FILLER = " Just so you know, I really appreciate you taking the time to go through all of this with me today."


def load(path: Path = DATA) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def as_turns(row: dict, pad: bool = False) -> list[tuple[str, str]]:
    return [
        (t["role"], t["content"] + (FILLER if pad and t["role"] == "trainee" else ""))
        for t in row["turns"]
    ]


async def _gather(coros, concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def run(c):
        async with sem:
            return await c

    return await asyncio.gather(*(run(c) for c in coros))


async def evaluate(
    judge, rows: list[dict], *, pairs: int = 40, concurrency: int = 8, seed: int = 3
) -> dict:
    predictions = await _gather([judge.score(as_turns(r)) for r in rows], concurrency)
    padded = await _gather([judge.score(as_turns(r, pad=True)) for r in rows], concurrency)

    per_dimension = {}
    per_phrasing: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"true": [], "pred": []})
    for d in DIMENSIONS:
        y_true = [r["labels"][d] for r in rows]
        y_pred = [p[d] for p in predictions]
        per_dimension[d] = agreement(y_true, y_pred)
        for r, p in zip(rows, predictions, strict=True):
            kind = r["phrasing"].get(d, "plain")
            per_phrasing[kind]["true"].append(r["labels"][d])
            per_phrasing[kind]["pred"].append(p[d])

    overall_true = [sum(r["labels"].values()) for r in rows]
    overall_pred = [sum(p.values()) for p in predictions]
    all_true = [r["labels"][d] for r in rows for d in DIMENSIONS]
    all_pred = [p[d] for p in predictions for d in DIMENSIONS]

    # Position bias: same pair in both orders.
    rng = random.Random(seed)
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scenario[r["scenario_id"]].append(r)
    pair_list = []
    while len(pair_list) < pairs:
        group = by_scenario[rng.choice(sorted(by_scenario))]
        a, b = rng.sample(group, 2)
        if sum(a["labels"].values()) != sum(b["labels"].values()):
            pair_list.append((a, b))
    forward = await _gather(
        [judge.compare(as_turns(a), as_turns(b)) for a, b in pair_list], concurrency
    )
    backward = await _gather(
        [judge.compare(as_turns(b), as_turns(a)) for a, b in pair_list], concurrency
    )
    flip = {"A": "B", "B": "A", "tie": "tie"}
    consistent = sum(f == flip[bk] for f, bk in zip(forward, backward, strict=True))
    correct = sum(
        f == ("A" if sum(a["labels"].values()) > sum(b["labels"].values()) else "B")
        for f, (a, b) in zip(forward, pair_list, strict=True)
    )
    verbosity_shift = sum(
        sum(p.values()) - sum(q.values()) for p, q in zip(padded, predictions, strict=True)
    ) / len(rows)

    return {
        "judge": judge.name,
        "n_transcripts": len(rows),
        "overall": {
            **agreement(all_true, all_pred),
            "spearman_total": spearman(overall_true, overall_pred),
        },
        "per_dimension": per_dimension,
        "per_phrasing": {
            k: {"n": len(v["true"]), **agreement(v["true"], v["pred"])}
            for k, v in sorted(per_phrasing.items())
        },
        "position_bias": {
            "pairs": len(pair_list),
            "consistent_rate": consistent / len(pair_list),
            "pairwise_accuracy": correct / len(pair_list),
        },
        "verbosity_bias": {"mean_total_score_shift": verbosity_shift},
    }


def to_markdown(report: dict) -> str:
    o = report["overall"]
    lines = [
        f"# Judge eval: {report['judge']}",
        "",
        f"{report['n_transcripts']} labelled transcripts x 5 dimensions.",
        "",
        f"- Exact agreement **{o['exact']:.1%}**, within one point {o['within_1']:.1%}, "
        f"quadratic weighted kappa **{o['qwk']:.2f}**, Spearman on total score {o['spearman_total']:.2f}",
        f"- Position consistency {report['position_bias']['consistent_rate']:.1%} over "
        f"{report['position_bias']['pairs']} swapped pairs; pairwise accuracy {report['position_bias']['pairwise_accuracy']:.1%}",
        f"- Verbosity probe: polite filler moves the total score by {report['verbosity_bias']['mean_total_score_shift']:+.2f} points on average",
        "",
        "| Dimension | Exact | Within 1 | QWK | Bias |",
        "|---|---|---|---|---|",
    ]
    for d, m in report["per_dimension"].items():
        lines.append(
            f"| {d} | {m['exact']:.1%} | {m['within_1']:.1%} | {m['qwk']:.2f} | {m['mean_bias']:+.2f} |"
        )
    lines += ["", "| Phrasing | n | Exact | QWK | Bias |", "|---|---|---|---|---|"]
    for k, m in report["per_phrasing"].items():
        lines.append(
            f"| {k} | {m['n']} | {m['exact']:.1%} | {m['qwk']:.2f} | {m['mean_bias']:+.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", choices=("keyword", "llm"), default="keyword")
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    judge = (
        KeywordJudge() if args.judge == "keyword" else LLMJudge(build_router(Settings.from_env()))
    )
    report = asyncio.run(evaluate(judge, load(), pairs=args.pairs, concurrency=args.concurrency))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", judge.name.lower()).strip("-")
    (out / f"eval_{slug}.json").write_text(json.dumps(report, indent=2))
    (out / f"eval_{slug}.md").write_text(to_markdown(report))
    print(to_markdown(report))


if __name__ == "__main__":
    main()
