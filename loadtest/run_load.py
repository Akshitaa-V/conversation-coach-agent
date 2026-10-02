"""Async load test: many concurrent roleplays over WebSocket, then feedback.

By default it starts the service in-process in mock mode and injects provider
failures, so it measures how the *system* behaves when a vendor misbehaves:
time to first token, full-reply latency, user-visible errors, fallbacks,
degraded replies and cost per conversation.

    python -m loadtest.run_load                                    # local, 20% primary failures
    python -m loadtest.run_load --url https://<service>.run.app    # against a deployment
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import socket
import statistics
import time
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import websockets

from coach.config import Settings

TRAINEE_LINES = [
    "Thanks for making the time. What does your current setup look like today?",
    "I hear you, and I'm sorry this happened.",
    "Help me understand, compared to what is the increase hard to justify?",
    "Teams your size usually save around 6 hours a week on manual reporting.",
    "Which part of this matters most to your team right now?",
    "Is it the price itself, or the timing within your budget year?",
    "Let's book 30 minutes next week, I'll send a calendar invite now.",
    "It has dashboards, alerts and an API.",
    "Let's stay in touch.",
]


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


async def one_conversation(
    client: httpx.AsyncClient, ws_base: str, idx: int, turns: int, rng: random.Random
) -> dict:
    out = {"ttft_ms": [], "reply_ms": [], "errors": 0, "degraded": 0, "fallbacks": 0, "replies": 0}
    r = await client.post(
        "/sessions",
        json={
            "trainee_id": f"load-{idx}",
            "scenario_id": rng.choice(
                ["renewal_price_increase", "fleet_discovery_call", "billing_complaint"]
            ),
        },
    )
    if r.status_code != 200:
        out["errors"] += 1
        return out
    session = r.json()
    out["variant"] = session["variant"]
    try:
        async with websockets.connect(
            f"{ws_base}/sessions/{session['session_id']}/stream", open_timeout=10
        ) as ws:
            for line in rng.sample(TRAINEE_LINES, turns):
                started = time.perf_counter()
                first = None
                await ws.send(json.dumps({"type": "message", "text": line}))
                while True:
                    event = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                    if event["type"] == "delta" and first is None:
                        first = time.perf_counter()
                    elif event["type"] == "end":
                        out["replies"] += 1
                        out["degraded"] += int(event["degraded"])
                        out["fallbacks"] += int(event["fallback"])
                        if first is not None:
                            out["ttft_ms"].append((first - started) * 1000)
                        out["reply_ms"].append((time.perf_counter() - started) * 1000)
                        break
                    elif event["type"] == "error":
                        out["errors"] += 1
                        break
            await ws.send(json.dumps({"type": "end_session"}))
    except (TimeoutError, OSError, websockets.WebSocketException):
        out["errors"] += 1
        return out

    started = time.perf_counter()
    fb = await client.post(f"/sessions/{session['session_id']}/feedback", timeout=60)
    out["feedback_ms"] = (time.perf_counter() - started) * 1000
    if fb.status_code != 200:
        out["errors"] += 1
        out["feedback_status"] = fb.status_code
        return out
    body = fb.json()
    out["cost_usd"] = body["cost_usd"]
    out["repairs"] = body["repairs"]
    return out


async def run(base_url: str, conversations: int, concurrency: int, turns: int, seed: int) -> dict:
    ws_base = base_url.replace("https://", "wss://").replace("http://", "ws://")
    rng = random.Random(seed)
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:

        async def guarded(i: int):
            async with sem:
                return await one_conversation(
                    client, ws_base, i, turns, random.Random(rng.random())
                )

        started = time.perf_counter()
        results = await asyncio.gather(*(guarded(i) for i in range(conversations)))
        wall_s = time.perf_counter() - started
        experiment = await client.get("/experiments/report")
        metrics_text = (await client.get("/metrics")).text

    ttft = [v for r in results for v in r["ttft_ms"]]
    reply = [v for r in results for v in r["reply_ms"]]
    feedback = [r["feedback_ms"] for r in results if "feedback_ms" in r]
    costs = [r["cost_usd"] for r in results if "cost_usd" in r]
    replies = sum(r["replies"] for r in results)
    failed_calls = sum(
        float(line.rsplit(" ", 1)[1])
        for line in metrics_text.splitlines()
        if line.startswith("coach_llm_requests_total{")
        and 'outcome="ok"' not in line
        and 'outcome="skipped_open"' not in line
    )
    return {
        "conversations": conversations,
        "concurrency": concurrency,
        "turns_per_conversation": turns,
        "wall_time_s": round(wall_s, 2),
        "replies": replies,
        "user_visible_errors": sum(r["errors"] for r in results),
        "degraded_replies": sum(r["degraded"] for r in results),
        "fallback_replies": sum(r["fallbacks"] for r in results),
        "failed_provider_calls": int(failed_calls),
        "feedback_repairs": sum(r.get("repairs", 0) for r in results),
        "ttft_ms": {"p50": _pct(ttft, 50), "p95": _pct(ttft, 95), "p99": _pct(ttft, 99)},
        "reply_ms": {"p50": _pct(reply, 50), "p95": _pct(reply, 95)},
        "feedback_ms": {"p50": _pct(feedback, 50), "p95": _pct(feedback, 95)},
        "cost_per_conversation_usd": {
            "mean": statistics.fmean(costs) if costs else 0.0,
            "p95": _pct(costs, 95),
        },
        "experiment_report": experiment.json() if experiment.status_code == 200 else None,
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def run_local(args) -> dict:
    import uvicorn

    from coach.app import create_app

    settings = replace(
        Settings.from_env(),
        db_path=args.db,
        mock_failure_rate={"mock/fast-a": args.failure_rate, "mock/smart-a": args.failure_rate},
        mock_bad_json_rate=args.bad_json_rate,
        mock_latency_ms=args.latency_ms,
        rate_limit_rpm=args.rate_limit_rpm,
        prioritise_realtime=not args.no_priority,
        breaker_reset_s=5.0,
        online_eval_sample_rate=0.2,
    )
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="warning")
    )
    task = asyncio.create_task(server.serve())
    while not server.started:  # noqa: ASYNC110 - uvicorn exposes a flag, not an event
        await asyncio.sleep(0.05)
    try:
        report = await run(
            f"http://127.0.0.1:{port}", args.conversations, args.concurrency, args.turns, args.seed
        )
    finally:
        server.should_exit = True
        await task
    report["injected"] = {
        "primary_failure_rate": args.failure_rate,
        "malformed_json_rate": args.bad_json_rate,
        "mock_latency_ms": args.latency_ms,
        "rate_limit_rpm": args.rate_limit_rpm,
        "realtime_priority": not args.no_priority,
    }
    return report


def to_markdown(r: dict) -> str:
    lines = [
        "# Load test",
        "",
        f"{r['conversations']} conversations x {r['turns_per_conversation']} turns, concurrency {r['concurrency']}, "
        f"{r['wall_time_s']} s wall time.",
    ]
    if "injected" in r:
        i = r["injected"]
        lines.append(
            f"Mock providers: {i['primary_failure_rate']:.0%} of primary-model calls fail (rate limit, timeout or 5xx), "
            f"{i['malformed_json_rate']:.0%} of coach answers are malformed, ~{i['mock_latency_ms']} ms base latency."
        )
    lines += [
        "",
        f"- Replies streamed: {r['replies']}; user-visible errors: **{r['user_visible_errors']}**; "
        f"degraded replies: {r['degraded_replies']}",
        f"- Failed provider calls absorbed by retries/fallbacks: {r['failed_provider_calls']}; "
        f"replies served by a fallback model: {r['fallback_replies']}; feedback repairs: {r['feedback_repairs']}",
        f"- Time to first token: p50 {r['ttft_ms']['p50']:.0f} ms, p95 **{r['ttft_ms']['p95']:.0f} ms**, p99 {r['ttft_ms']['p99']:.0f} ms",
        f"- Full reply: p50 {r['reply_ms']['p50']:.0f} ms, p95 {r['reply_ms']['p95']:.0f} ms; "
        f"feedback: p50 {r['feedback_ms']['p50']:.0f} ms, p95 {r['feedback_ms']['p95']:.0f} ms",
        f"- Cost per conversation (illustrative mock prices): mean ${r['cost_per_conversation_usd']['mean']:.5f}, "
        f"p95 ${r['cost_per_conversation_usd']['p95']:.5f}",
    ]
    exp = r.get("experiment_report")
    if exp:
        lines += [
            "",
            "## A/B readout on this traffic (control vs realism_v2)",
            "",
            f"- n = {exp['n_control']} / {exp['n_treatment']}; score diff {exp['diff']:+.3f} "
            f"(95% CI {exp['diff_ci'][0]:+.3f} to {exp['diff_ci'][1]:+.3f}); cost {exp['guardrail_rel_change']:+.1%} "
            f"(95% CI {exp['guardrail_rel_ci'][0]:+.1%} to {exp['guardrail_rel_ci'][1]:+.1%})",
            f"- Decision: **{exp['decision']}** ({exp['reason']})",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="target a running service instead of starting one locally")
    parser.add_argument("--conversations", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--failure-rate", type=float, default=0.2)
    parser.add_argument("--bad-json-rate", type=float, default=0.1)
    parser.add_argument("--latency-ms", type=int, default=60)
    parser.add_argument(
        "--rate-limit-rpm", type=int, default=6000, help="per-vendor token bucket (local runs)"
    )
    parser.add_argument(
        "--no-priority",
        action="store_true",
        help="one FIFO queue for all tasks instead of realtime-first",
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--db", default="loadtest.db")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    import logging

    from coach.telemetry import setup_logging

    setup_logging("WARNING")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("coach").setLevel(logging.ERROR)  # injected failures are expected here
    if not args.url:
        Path(args.db).unlink(missing_ok=True)
    report = asyncio.run(
        run(args.url, args.conversations, args.concurrency, args.turns, args.seed)
        if args.url
        else run_local(args)
    )
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "load_test.json").write_text(json.dumps(report, indent=2))
    (out / "load_test.md").write_text(to_markdown(report))
    print(to_markdown(report))


if __name__ == "__main__":
    main()
