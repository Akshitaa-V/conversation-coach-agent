# Judge eval: keyword-baseline

90 labelled transcripts x 5 dimensions.

- Exact agreement **66.7%**, within one point 66.7%, quadratic weighted kappa **0.50**, Spearman on total score 0.54
- Position consistency 100.0% over 40 swapped pairs; pairwise accuracy 70.0%
- Verbosity probe: polite filler moves the total score by +0.00 points on average

| Dimension | Exact | Within 1 | QWK | Bias |
|---|---|---|---|---|
| discovery | 58.9% | 58.9% | 0.62 | -0.38 |
| empathy | 52.2% | 52.2% | 0.16 | -0.20 |
| value | 73.3% | 73.3% | 0.62 | -0.71 |
| objection_handling | 72.2% | 72.2% | 0.32 | +0.19 |
| next_step | 76.7% | 76.7% | 0.70 | -0.62 |

| Phrasing | n | Exact | QWK | Bias |
|---|---|---|---|---|
| decoy | 29 | 20.7% | 0.11 | +1.90 |
| explicit | 186 | 95.2% | 0.91 | +0.04 |
| implicit | 102 | 5.9% | 0.08 | -2.73 |
| plain | 133 | 83.5% | 0.00 | +0.45 |
