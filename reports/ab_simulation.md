# A/B pipeline check on planted effects

Simulated session data with known ground truth. Score = mean feedback score (1-5); guardrail = LLM cost per conversation.

| Case | Planted lift | Planted cost change | Diff (95% CI) | Cost change (95% CI) | SRM p | Decision | Expected |
|---|---|---|---|---|---|---|---|
| planted_effect_cheap | +0.25 | +4% | +0.252 (+0.203, +0.302) | +5.1% (+3.2%, +7.2%) | 1 | **ship** | ship |
| planted_effect_costly | +0.25 | +30% | +0.278 (+0.229, +0.330) | +29.2% (+26.8%, +31.7%) | 1 | **hold** | hold |
| no_effect_costly | +0.00 | +25% | -0.025 (-0.071, +0.018) | +25.7% (+23.7%, +27.6%) | 1 | **kill** | kill |
| too_few_sessions | +0.12 | +0% | -0.154 (-0.400, +0.086) | +4.8% (-4.1%, +14.4%) | 1 | **keep_running** | keep_running |
| broken_assignment | +0.25 | +4% | +0.271 (+0.226, +0.314) | +4.7% (+3.1%, +6.3%) | <0.001 | **invalid** | invalid |

Correct decisions: 5/5

## Calibration

- A/A runs (400 x 1000 per arm): false ship rate **2.5%** (a two-sided 95% CI should ship by chance in about 2.5% of runs)
- Planted lift of 0.15 : 95% CI covered the true difference in **94.8%** of runs; effect detected in 99.2%
