# Load test

200 conversations x 4 turns, concurrency 40, 9.54 s wall time.
Mock providers: 20% of primary-model calls fail (rate limit, timeout or 5xx), 10% of coach answers are malformed, ~60 ms base latency.

- Replies streamed: 800; user-visible errors: **0**; degraded replies: 0
- Failed provider calls absorbed by retries/fallbacks: 261; replies served by a fallback model: 153; feedback repairs: 17
- Time to first token: p50 83 ms, p95 **171 ms**, p99 210 ms
- Full reply: p50 162 ms, p95 264 ms; feedback: p50 706 ms, p95 2783 ms
- Cost per conversation (illustrative mock prices): mean $0.00440, p95 $0.00689

## A/B readout on this traffic (control vs realism_v2)

- n = 85 / 115; score diff +0.007 (95% CI -0.130 to +0.140); cost +3.0% (95% CI -2.7% to +8.9%)
- Decision: **keep_running** (below the planned 369 units per arm; do not decide yet)
