"""Keyword-based rubric scorer.

This is deliberately simple. It is the *baseline* judge in the evals (the bar an
LLM judge has to beat) and the scripted brain of the offline mock coach. It only
recognises explicit phrasings, so it misses implicit ones by design; the eval
report shows exactly how much that costs.
"""

from __future__ import annotations

import re

from coach.schemas import DIMENSIONS

_OPEN_Q = re.compile(r"\b(what|how|why|which|tell me|walk me through)\b[^?]*\?", re.I)
_EMPATHY = re.compile(
    r"\b(i understand|i hear you|that sounds (frustrating|stressful|annoying)|i'm sorry|i am sorry|"
    r"that must be|i get why|completely understand|fair concern)\b",
    re.I,
)
_GENERIC_EMPATHY = re.compile(r"\b(okay|i see|got it|noted)\b", re.I)
_QUANTIFIED = re.compile(r"\b\d+(\.\d+)?\s?(%|percent|hours?|days?|euros?|eur|k\b|minutes?)", re.I)
_BENEFIT = re.compile(r"\b(save|saves|saving|reduce|reduces|cut|cuts|lower|faster|improve)\b", re.I)
_OBJ_EXPLORE = re.compile(
    r"\b(what would|compared to what|help me understand|what matters most|if we could|"
    r"what if|is it the price itself|apart from)\b",
    re.I,
)
_OBJ_DISMISS = re.compile(
    r"\b(that's just how it is|nothing i can do|you're wrong|that is not true)\b", re.I
)
_DATED_STEP = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|tomorrow|next week|\d{1,2}(:\d{2})?\s?(am|pm)|"
    r"by end of day|calendar invite)\b",
    re.I,
)
_VAGUE_STEP = re.compile(
    r"\b(stay in touch|follow up|get back to you|talk soon|send (you )?(some )?info)\b", re.I
)


def _first(pattern: re.Pattern[str], turns: list[str]) -> str:
    for turn in turns:
        if pattern.search(turn):
            return turn
    return ""


def keyword_scores(trainee_turns: list[str]) -> dict[str, tuple[int, str]]:
    """Return {dimension: (score 1-5, evidence turn)} from trainee turns only."""
    text_turns = [t.strip() for t in trainee_turns if t.strip()]
    out: dict[str, tuple[int, str]] = {}

    questions = [t for t in text_turns if _OPEN_Q.search(t)]
    out["discovery"] = (
        5 if len(questions) >= 2 else 3 if questions else 1,
        questions[0] if questions else "",
    )

    ev = _first(_EMPATHY, text_turns)
    if ev:
        out["empathy"] = (5, ev)
    else:
        ev = _first(_GENERIC_EMPATHY, text_turns)
        out["empathy"] = (3 if ev else 1, ev)

    quant = [t for t in text_turns if _QUANTIFIED.search(t) and _BENEFIT.search(t)]
    if quant:
        out["value"] = (5, quant[0])
    else:
        ev = _first(_BENEFIT, text_turns)
        out["value"] = (3 if ev else 1, ev)

    ev = _first(_OBJ_EXPLORE, text_turns)
    if ev:
        out["objection_handling"] = (5, ev)
    else:
        ev = _first(_OBJ_DISMISS, text_turns)
        out["objection_handling"] = (2 if ev else 1, ev)

    ev = _first(_DATED_STEP, text_turns)
    if ev:
        out["next_step"] = (5, ev)
    else:
        ev = _first(_VAGUE_STEP, text_turns)
        out["next_step"] = (3 if ev else 1, ev)

    assert set(out) == set(DIMENSIONS)
    return out
