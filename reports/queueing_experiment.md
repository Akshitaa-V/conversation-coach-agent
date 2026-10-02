# Rate-limit queueing experiment

200 conversations x 4 turns at concurrency 40, 6000 requests/min per vendor, no injected failures, 3 runs per arm (mean shown).

| Policy | TTFT p50 | TTFT p95 | Feedback p95 | Errors |
|---|---|---|---|---|
| fifo | 82 ms | 388 ms | 762 ms | 0 |
| realtime_first | 76 ms | 131 ms | 1683 ms | 0 |
