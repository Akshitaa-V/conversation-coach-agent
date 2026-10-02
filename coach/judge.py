"""Judges that score a finished transcript against the rubric.

``KeywordJudge`` is the baseline. ``LLMJudge`` asks a model to score with the
same rubric, and also supports a pairwise mode used to measure position bias
(does the verdict flip when the two transcripts swap places?).
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from coach.agents.coach import _extract_json, format_transcript
from coach.llm.router import ModelRouter
from coach.scenarios import RUBRIC
from coach.schemas import DIMENSIONS, JudgeScores, PairwiseVerdict
from coach.scoring import keyword_scores

Turns = list[tuple[str, str]]


class KeywordJudge:
    name = "keyword-baseline"

    async def score(self, turns: Turns) -> dict[str, int]:
        scores = keyword_scores([text for role, text in turns if role == "trainee"])
        return {d: scores[d][0] for d in DIMENSIONS}

    async def compare(self, a: Turns, b: Turns) -> str:
        sa, sb = sum((await self.score(a)).values()), sum((await self.score(b)).values())
        return "A" if sa > sb else "B" if sb > sa else "tie"


_RUBRIC_TEXT = "\n".join(f"- {d}: {RUBRIC[d]}" for d in DIMENSIONS)


class LLMJudge:
    def __init__(self, router: ModelRouter, task: str = "judge") -> None:
        self.router = router
        self.task = task
        self.name = "llm:" + ",".join(router.routes[task])

    async def score(self, turns: Turns) -> dict[str, int]:
        messages = [
            {
                "role": "system",
                "content": "[task:judge]\nYou grade sales and service practice calls. Score only the Trainee, "
                "using this rubric:\n" + _RUBRIC_TEXT + "\n"
                'Answer with JSON only: {"discovery": 1-5, "empathy": 1-5, "value": 1-5, '
                '"objection_handling": 1-5, "next_step": 1-5}',
            },
            {"role": "user", "content": "Transcript:\n" + format_transcript(turns)},
        ]
        for _ in range(2):  # one retry on unparsable output
            completion = await self.router.complete(
                self.task, messages, json_mode=True, temperature=0.0
            )
            try:
                return JudgeScores.model_validate_json(_extract_json(completion.text)).as_dict()
            except (ValidationError, ValueError):
                messages.append({"role": "assistant", "content": completion.text})
                messages.append(
                    {"role": "user", "content": "Invalid. Answer with the JSON object only."}
                )
        raise ValueError("judge returned invalid JSON twice")

    async def compare(self, a: Turns, b: Turns) -> str:
        messages = [
            {
                "role": "system",
                "content": "[task:judge_pairwise]\nTwo trainees handled the same customer. Using the rubric below, "
                "decide who handled the call better overall.\n" + _RUBRIC_TEXT + "\n"
                'Answer with JSON only: {"winner": "A" | "B" | "tie"}',
            },
            {
                "role": "user",
                "content": "### Transcript A\n"
                + format_transcript(a)
                + "\n\n### Transcript B\n"
                + format_transcript(b),
            },
        ]
        completion = await self.router.complete(
            "judge_pairwise" if "judge_pairwise" in self.router.routes else self.task,
            messages,
            json_mode=True,
            temperature=0.0,
        )
        try:
            return PairwiseVerdict.model_validate(json.loads(_extract_json(completion.text))).winner
        except (ValidationError, ValueError):
            return "tie"
