"""Coach agent: reads a finished roleplay and returns validated, grounded feedback.

Flow: the model may call ``get_rubric`` / ``find_examples`` (bounded number of
steps), then must answer with JSON matching ``schemas.Feedback``. The answer is
checked twice before it reaches a trainee:

1. schema validation (Pydantic): five dimensions, scores 1-5, length limits;
2. grounding: every ``evidence`` quote must actually appear in a trainee turn,
   so the coach cannot praise something the trainee never said.

A failed check is sent back to the model once as a repair request. If the
repair fails too, the call raises and the API answers 503 instead of showing
unchecked feedback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from coach import telemetry as tm
from coach.agents.tools import TOOL_SCHEMAS, run_tool
from coach.llm.base import LLMError, Message
from coach.llm.router import ModelRouter
from coach.scenarios import Scenario
from coach.schemas import Feedback

MAX_TOOL_STEPS = 4
MAX_REPAIRS = 1


class FeedbackInvalid(LLMError):
    retryable = False


@dataclass
class CoachResult:
    feedback: Feedback
    model: str
    repairs: int
    tool_calls: int


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip(" .!?\"'")


def ungrounded_evidence(feedback: Feedback, trainee_turns: list[str]) -> list[str]:
    turns = [_norm(t) for t in trainee_turns]
    return [
        s.dimension
        for s in feedback.scores
        if s.evidence.strip() and not any(_norm(s.evidence) in t for t in turns)
    ]


def format_transcript(turns: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"{'Trainee' if role == 'trainee' else 'Customer'}: {text}" for role, text in turns
    )


def system_prompt(scenario: Scenario, focus_areas: list[str]) -> str:
    focus = (
        f"From earlier sessions this trainee most needs work on: {', '.join(focus_areas)}. "
        "Say whether they improved on these.\n"
        if focus_areas
        else ""
    )
    return (
        "[task:coach]\n"
        f"You are a sales and service coach reviewing a practice call (scenario: {scenario.id}).\n"
        "Use the tools to read the rubric and, if useful, guidance for the weakest dimension. "
        f"{focus}"
        "Then answer with JSON only, in this shape:\n"
        '{"scores": [{"dimension": "discovery|empathy|value|objection_handling|next_step", '
        '"score": 1-5, "evidence": "exact quote from a Trainee line, or empty"}], '
        '"strengths": ["max 3 short items"], "next_focus": "one concrete thing to practise next"}\n'
        "Score all five dimensions exactly once. Quote evidence word for word from Trainee lines only."
    )


class CoachAgent:
    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    async def review(
        self,
        scenario: Scenario,
        turns: list[tuple[str, str]],
        *,
        focus_areas: list[str] | None = None,
        conversation_id: str = "",
    ) -> CoachResult:
        trainee_turns = [text for role, text in turns if role == "trainee"]
        messages: list[Message] = [
            {"role": "system", "content": system_prompt(scenario, focus_areas or [])},
            {"role": "user", "content": "Transcript:\n" + format_transcript(turns)},
        ]
        tool_calls = 0
        repairs = 0
        steps = 0
        while True:
            use_tools = steps < MAX_TOOL_STEPS
            completion = await self.router.complete(
                "coach",
                messages,
                conversation_id=conversation_id,
                tools=TOOL_SCHEMAS if use_tools else None,
                json_mode=not use_tools,
                temperature=0.2,
            )
            steps += 1
            if completion.tool_calls and use_tools:
                messages.append(
                    {
                        "role": "assistant",
                        "content": completion.text or None,
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": c.arguments},
                            }
                            for c in completion.tool_calls
                        ],
                    }
                )
                for call in completion.tool_calls:
                    tool_calls += 1
                    with tm.span("tool." + call.name, conversation_id=conversation_id):
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": run_tool(call.name, call.arguments),
                            }
                        )
                continue

            problem = self._check(completion.text, trainee_turns)
            if problem is None:
                feedback = Feedback.model_validate_json(_extract_json(completion.text))
                return CoachResult(feedback, completion.model, repairs, tool_calls)

            reason, detail = problem
            tm.FEEDBACK_REPAIRS.labels(reason).inc()
            if repairs >= MAX_REPAIRS:
                raise FeedbackInvalid(f"feedback still invalid after {repairs} repair(s): {detail}")
            repairs += 1
            messages.append({"role": "assistant", "content": completion.text})
            messages.append(
                {
                    "role": "user",
                    "content": f"Your previous answer was invalid ({reason}): {detail}. "
                    "Answer again with corrected JSON only.",
                }
            )

    @staticmethod
    def _check(text: str, trainee_turns: list[str]) -> tuple[str, str] | None:
        try:
            feedback = Feedback.model_validate_json(_extract_json(text))
        except (ValidationError, ValueError) as exc:
            return "schema", str(exc).splitlines()[0][:300]
        missing = ungrounded_evidence(feedback, trainee_turns)
        if missing:
            return "ungrounded", f"evidence for {missing} is not a quote from a Trainee line"
        return None


def _extract_json(text: str) -> str:
    """Accept bare JSON or JSON wrapped in a Markdown code fence."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the answer")
    candidate = text[start : end + 1]
    json.loads(candidate)
    return candidate
