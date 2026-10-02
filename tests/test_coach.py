from __future__ import annotations

import json

import pytest

from coach.agents.coach import CoachAgent, FeedbackInvalid, ungrounded_evidence
from coach.agents.tools import find_examples, run_tool
from coach.llm.base import Completion
from coach.scenarios import get_scenario
from coach.schemas import DIMENSIONS, Feedback

TURNS = [
    ("customer", "Honestly, a 12% increase is hard to justify to my CFO."),
    ("trainee", "I hear you, and I'm sorry this happened."),
    ("trainee", "What does your current setup look like today?"),
    ("trainee", "Is it the price itself, or the timing within your budget year?"),
    ("trainee", "Teams your size usually save around 6 hours a week on manual reporting."),
    ("trainee", "Let's book 30 minutes next week, I'll send a calendar invite now."),
]


async def test_coach_uses_tools_and_returns_valid_feedback(make_router):
    router, _ = make_router()
    result = await CoachAgent(router).review(get_scenario("renewal_price_increase"), TURNS)
    assert result.tool_calls == 2
    assert result.repairs == 0
    assert {s.dimension for s in result.feedback.scores} == set(DIMENSIONS)
    assert result.feedback.score_of("next_step") == 5
    assert not ungrounded_evidence(result.feedback, [t for r, t in TURNS if r == "trainee"])


async def test_coach_repairs_schema_violation(make_router):
    router, _ = make_router(bad_json_rate=1.0)
    result = await CoachAgent(router).review(get_scenario("renewal_price_increase"), TURNS)
    assert result.repairs == 1
    assert len(result.feedback.scores) == 5


class _ScriptedCoachProvider:
    """Returns the same final answer every time, to test the repair limit and grounding check."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def complete(self, model, messages, *, tools=None, **kwargs):
        return Completion(text=json.dumps(self.payload), model=model)


def _payload(evidence: str) -> dict:
    return {
        "scores": [
            {"dimension": d, "score": 3, "evidence": evidence if d == "empathy" else ""}
            for d in DIMENSIONS
        ],
        "strengths": [],
        "next_focus": "Ask more questions.",
    }


async def test_hallucinated_evidence_is_rejected(make_router):
    router, _ = make_router()
    router.providers["fake"] = _ScriptedCoachProvider(_payload("I completely understand your pain"))
    router.routes["coach"] = ("fake/model",)
    with pytest.raises(FeedbackInvalid, match="evidence"):
        await CoachAgent(router).review(get_scenario("renewal_price_increase"), TURNS)


async def test_grounded_evidence_passes_with_case_and_punctuation_differences(make_router):
    router, _ = make_router()
    router.providers["fake"] = _ScriptedCoachProvider(
        _payload("i hear you, and I'm sorry this happened")
    )
    router.routes["coach"] = ("fake/model",)
    result = await CoachAgent(router).review(get_scenario("renewal_price_increase"), TURNS)
    assert result.feedback.score_of("empathy") == 3


def test_feedback_requires_every_dimension_once():
    with pytest.raises(ValueError):
        Feedback.model_validate(
            {
                "scores": [{"dimension": "empathy", "score": 3}] * 5,
                "strengths": [],
                "next_focus": "x" * 5,
            }
        )


def test_tool_errors_go_back_to_the_model_as_data():
    assert "error" in json.loads(run_tool("get_rubric", json.dumps({"scenario_id": "nope"})))
    assert "error" in json.loads(run_tool("delete_everything", "{}"))
    assert "error" in json.loads(run_tool("find_examples", "{not json"))


def test_find_examples_filters_by_dimension():
    examples = find_examples("objection_handling", "competitor quote is cheaper")["examples"]
    assert any("competitor" in e for e in examples)
