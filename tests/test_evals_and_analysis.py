from __future__ import annotations

import numpy as np
import pytest

from coach.judge import KeywordJudge, LLMJudge
from evals.metrics import agreement, quadratic_weighted_kappa
from evals.run_eval import evaluate, load
from experiments.analysis import analyse, required_n_per_arm, srm_p_value


def _arms(rng, n, lift, cost_lift):
    scores = np.clip(rng.normal(3.1 + lift, 0.7, n), 1, 5)
    costs = rng.lognormal(np.log(0.004 * (1 + cost_lift)), 0.25, n)
    return scores, costs


@pytest.mark.parametrize(
    ("n", "lift", "cost_lift", "expected"),
    [
        (1500, 0.25, 0.03, "ship"),
        (1500, 0.25, 0.30, "hold"),
        (2000, -0.05, 0.2, "kill"),
        (50, 0.2, 0.0, "keep_running"),
    ],
)
# A pure null (lift 0) is not used here: about 2.5% of seeds produce a CI above zero by chance,
# which is exactly the false-positive rate experiments/simulate_ab.py measures over 400 runs.
def test_decision_rule_on_planted_effects(n, lift, cost_lift, expected):
    rng = np.random.default_rng(42)
    a, ga = _arms(rng, n, 0, 0)
    b, gb = _arms(rng, n, lift, cost_lift)
    assert analyse(a, b, ga, gb).decision == expected


def test_sample_ratio_mismatch_blocks_the_readout():
    rng = np.random.default_rng(1)
    a, ga = _arms(rng, 2200, 0, 0)
    b, gb = _arms(rng, 1800, 0.3, 0)
    report = analyse(a, b, ga, gb)
    assert report.decision == "invalid"
    assert srm_p_value(1000, 1000) == pytest.approx(1.0)


def test_required_sample_size_matches_textbook_formula():
    # 2 * (1.96 + 0.84)^2 * sd^2 / delta^2 with sd = 1, delta = 0.5 -> about 63
    assert required_n_per_arm(1.0, 0.5) in (63, 64)


def test_qwk_extremes():
    assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert quadratic_weighted_kappa([1, 5, 1, 5], [5, 1, 5, 1]) < 0
    m = agreement([1, 3, 5], [2, 3, 5])
    assert m["exact"] == pytest.approx(2 / 3) and m["within_1"] == 1.0 and m["mean_bias"] > 0


async def test_keyword_judge_eval_shows_where_it_breaks():
    report = await evaluate(KeywordJudge(), load(), pairs=10)
    phr = report["per_phrasing"]
    assert phr["explicit"]["exact"] > 0.9  # catches textbook wording
    assert phr["implicit"]["exact"] < 0.3  # misses natural wording
    assert phr["decoy"]["mean_bias"] > 0  # fooled by keyword-heavy decoys
    assert report["position_bias"]["consistent_rate"] == 1.0


async def test_llm_judge_plumbing_with_mock_router(make_router):
    router, _ = make_router()
    rows = load()[:12]
    report = await evaluate(LLMJudge(router), rows, pairs=4)
    assert report["n_transcripts"] == 12
    assert 0 <= report["overall"]["exact"] <= 1
