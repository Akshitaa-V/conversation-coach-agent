"""Scripted responders that give the mock provider plausible behaviour per task.

The task is recognised from a tag in the system prompt (``[task:customer]`` etc.),
which the agents add anyway so traces are easy to filter.
"""

from __future__ import annotations

import json
import random
import re

from coach.llm.base import Completion, Message, ToolCall
from coach.scenarios import SCENARIOS
from coach.schemas import DIMENSIONS
from coach.scoring import keyword_scores

_TASK = re.compile(r"\[task:([a-z_]+)\]")
_SCENARIO = re.compile(r"\(scenario: ([a-z_]+)\)")
_VARIANT = re.compile(r"\(variant: ([a-z0-9_]+)\)")
_TRAINEE_LINE = re.compile(r"^Trainee: (.*)$", re.M)


def _system(messages: list[Message]) -> str:
    return next((m["content"] for m in messages if m["role"] == "system"), "")


def _task(messages: list[Message]) -> str:
    match = _TASK.search(_system(messages))
    return match.group(1) if match else "chat"


def _customer(messages: list[Message]) -> Completion:
    system = _system(messages)
    scenario = SCENARIOS[_SCENARIO.search(system).group(1)]
    variant_match = _VARIANT.search(system)
    variant = variant_match.group(1) if variant_match else "control"
    trainee_turns = [m for m in messages if m["role"] == "user"]
    n = len(trainee_turns)
    rng = random.Random(
        f"{scenario.id}:{n}:{trainee_turns[-1]['content'] if trainee_turns else ''}"
    )
    if n % 2 == 1 and (n // 2) < len(scenario.objections):
        reply = scenario.objections[n // 2]
    else:
        reply = rng.choice(scenario.reactions)
    if variant == "realism_v2":
        # The treatment prompt asks for more natural, slightly longer replies.
        reply = f"{reply} {rng.choice(('Let me think about that for a second.', 'To be fair, I need to check with my team.', 'Just so you know where I stand.'))}"
    return Completion(text=reply, model="")


def _transcript_trainee_turns(messages: list[Message]) -> list[str]:
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return [t.strip() for t in _TRAINEE_LINE.findall(user)]


def _coach(messages: list[Message], tools: list[dict] | None, bad_json: bool) -> Completion:
    has_tool_results = any(m["role"] == "tool" for m in messages)
    if tools and not has_tool_results:
        scenario = _SCENARIO.search(_system(messages))
        sid = scenario.group(1) if scenario else "renewal_price_increase"
        return Completion(
            text="",
            model="",
            tool_calls=[
                ToolCall(
                    id="call_1", name="get_rubric", arguments=json.dumps({"scenario_id": sid})
                ),
                ToolCall(
                    id="call_2",
                    name="find_examples",
                    arguments=json.dumps(
                        {"dimension": "objection_handling", "query": "price objection"}
                    ),
                ),
            ],
        )
    repairing = any("previous answer was invalid" in (m.get("content") or "") for m in messages)
    scores = keyword_scores(_transcript_trainee_turns(messages))
    payload = {
        "scores": [
            {"dimension": d, "score": scores[d][0], "evidence": scores[d][1][:300]}
            for d in DIMENSIONS
        ],
        "strengths": [d.replace("_", " ") for d in DIMENSIONS if scores[d][0] >= 4][:3],
        "next_focus": "Work on "
        + min(DIMENSIONS, key=lambda d: scores[d][0]).replace("_", " ")
        + " in the next call.",
    }
    if bad_json and not repairing:
        payload["scores"] = payload["scores"][:3]  # schema violation the agent must repair
    return Completion(text=json.dumps(payload), model="")


def _judge(messages: list[Message]) -> Completion:
    scores = keyword_scores(_transcript_trainee_turns(messages))
    return Completion(text=json.dumps({d: scores[d][0] for d in DIMENSIONS}), model="")


def _judge_pairwise(messages: list[Message]) -> Completion:
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    part_a, _, part_b = user.partition("### Transcript B")
    total_a = sum(s for s, _ in keyword_scores(_TRAINEE_LINE.findall(part_a)).values())
    total_b = sum(s for s, _ in keyword_scores(_TRAINEE_LINE.findall(part_b)).values())
    winner = "A" if total_a > total_b else "B" if total_b > total_a else "tie"
    return Completion(text=json.dumps({"winner": winner}), model="")


def make_responder(bad_json_rate: float = 0.0, seed: int = 11):
    rng = random.Random(seed)

    def respond(
        model: str, messages: list[Message], tools: list[dict] | None, json_mode: bool
    ) -> Completion:
        task = _task(messages)
        if task == "customer":
            return _customer(messages)
        if task == "coach":
            return _coach(messages, tools, bad_json=rng.random() < bad_json_rate)
        if task == "judge":
            return _judge(messages)
        if task == "judge_pairwise":
            return _judge_pairwise(messages)
        return Completion(text="OK.", model="")

    return respond
