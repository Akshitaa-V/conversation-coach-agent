"""Build the labelled eval set: roleplay transcripts with planted ground truth.

Each transcript is assembled from trainee behaviours chosen per rubric
dimension (e.g. discovery = 0, 1 or 2 open questions), so the correct score is
known by construction. Every behaviour has three kinds of phrasing:

* ``explicit``: textbook wording a keyword rule can catch;
* ``implicit``: the same behaviour in natural wording with no tell-tale keyword;
* ``decoy``: keyword-heavy wording that does *not* show the behaviour
  (e.g. "I understand, but that's just how pricing works").

That mix is what separates a judge that understands the call from one that
pattern-matches. The set is synthetic; real human-labelled transcripts should
be added alongside it before trusting any judge in production.

    python -m evals.build_dataset   # writes evals/data/transcripts.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from coach.scenarios import SCENARIOS

# ------------------------------------------------------------------ phrasing pools
DISCOVERY_Q = {
    "explicit": [
        "What does your current setup look like today?",
        "How are you handling this at the moment?",
        "What would a good outcome look like for you this year?",
        "Which part of this matters most to your team right now?",
    ],
    "implicit": [
        "Before I suggest anything, I'd love to hear how things run on your side today.",
        "Talk me through a normal week for your team, I want to understand where it hurts.",
        "I'm curious where this sits on your list of priorities for the year.",
        "Give me a sense of what success would mean for you here.",
    ],
}
EMPATHY = {
    5: {
        "explicit": [
            "I understand, that sounds frustrating.",
            "I hear you, and I'm sorry this happened.",
            "That must be stressful with a fixed budget.",
        ],
        "implicit": [
            "Forty minutes on hold after a mistake that wasn't yours, that is a rough start to the day.",
            "Having to defend a bigger number to your CFO is not a fun conversation, fair enough.",
            "If the last tool annoyed your drivers, of course you are wary of the next one.",
        ],
    },
    3: {
        "explicit": ["Okay, noted.", "Got it.", "I see."],
        "decoy": [
            "I understand, but that's how our pricing works.",
            "I hear you. Anyway, let me continue.",
        ],
    },
    1: {"plain": ["Right, so let me tell you about our offer.", "Let me go through the details."]},
}
VALUE = {
    5: {
        "explicit": [
            "Teams your size usually save around 6 hours a week on manual reporting.",
            "The premium plan cuts unplanned outages by about 30 percent based on last year's data.",
            "That would reduce your refund handling time to under 2 days.",
        ],
        "implicit": [
            "Most fleets like yours get roughly a full working day back every week once dispatch runs itself.",
            "With the redundant line, the two outages you had last quarter would not have reached your users at all.",
            "Over the three years you would come out about 4,000 euros ahead of the cheaper quote.",
        ],
    },
    3: {
        "explicit": [
            "It will improve your team's efficiency.",
            "This should make things faster for everyone.",
        ],
        "implicit": [
            "Overall it just makes the day-to-day smoother.",
            "People generally find it worth it.",
        ],
    },
    1: {
        "plain": [
            "It has dashboards, alerts and an API.",
            "The package includes 24/7 support and a portal.",
        ]
    },
}
OBJECTION = {
    5: {
        "explicit": [
            "Help me understand, compared to what is the increase hard to justify?",
            "What would need to be true for the price to make sense to your CFO?",
            "Is it the price itself, or the timing within your budget year?",
        ],
        "implicit": [
            "Let's look at that quote together; does it include the same uptime guarantee and on-site support?",
            "Is the concern the number on its own, or that it lands in the middle of your budget year?",
            "Walk me through what the competitor's offer covers, so we compare the same thing.",
        ],
    },
    2: {
        "explicit": [
            "Well, that's just how it is this year.",
            "There is nothing I can do about the price.",
        ],
        "decoy": [
            "I understand, but that's just how it is with all providers.",
            "What if I told you everyone pays this? It's the market rate.",
        ],
    },
    1: {"plain": ["Anyway, moving on to the next point.", "Let me continue with the features."]},
}
NEXT_STEP = {
    5: {
        "explicit": [
            "I'll send the comparison by Thursday 10 am and we review it together on Friday.",
            "Let's book 30 minutes next week, I'll send a calendar invite now.",
            "I'll process the refund today and call you tomorrow to confirm it arrived.",
        ],
        "implicit": [
            "I'll put us both down for the 14th at half past nine and bring the numbers.",
            "You'll have the written offer from me before your Thursday board meeting, and I'll ring you that afternoon.",
            "I'll get the refund moving before we hang up and check back with you the morning after.",
        ],
    },
    3: {
        "explicit": [
            "Let's stay in touch.",
            "I'll follow up at some point.",
            "I'll get back to you.",
        ],
        "implicit": [
            "Have a think and we can pick it up again.",
            "Let me know whenever you're ready.",
        ],
    },
    1: {"plain": ["Thanks for your time.", "Okay, that's all from my side."]},
}

LEVELS = {
    "discovery": {0: 1, 1: 3, 2: 5},  # number of open questions -> score
    "empathy": (5, 3, 1),
    "value": (5, 3, 1),
    "objection_handling": (5, 2, 1),
    "next_step": (5, 3, 1),
}


def _pick(pool: dict, rng: random.Random, implicit_share: float) -> tuple[str, str]:
    kinds = list(pool)
    if "plain" in kinds:
        return rng.choice(pool["plain"]), "plain"
    weights = []
    for kind in kinds:
        weights.append(implicit_share if kind in ("implicit", "decoy") else 1 - implicit_share)
    kind = rng.choices(kinds, weights=weights)[0]
    return rng.choice(pool[kind]), kind


def build(n: int = 90, seed: int = 4242, implicit_share: float = 0.4) -> list[dict]:
    rng = random.Random(seed)
    scenarios = list(SCENARIOS.values())
    rows = []
    for i in range(n):
        scenario = scenarios[i % len(scenarios)]
        n_questions = rng.choice((0, 1, 2))
        levels = {
            "discovery": LEVELS["discovery"][n_questions],
            "empathy": rng.choice(LEVELS["empathy"]),
            "value": rng.choice(LEVELS["value"]),
            "objection_handling": rng.choice(LEVELS["objection_handling"]),
            "next_step": rng.choice(LEVELS["next_step"]),
        }
        kinds: dict[str, str] = {}
        questions = []
        for _ in range(n_questions):
            text, kind = _pick(DISCOVERY_Q, rng, implicit_share)
            while text in questions:
                text, kind = _pick(DISCOVERY_Q, rng, implicit_share)
            questions.append(text)
            kinds.setdefault("discovery", kind)
        empathy, kinds["empathy"] = _pick(EMPATHY[levels["empathy"]], rng, implicit_share)
        value, kinds["value"] = _pick(VALUE[levels["value"]], rng, implicit_share)
        objection, kinds["objection_handling"] = _pick(
            OBJECTION[levels["objection_handling"]], rng, implicit_share
        )
        step, kinds["next_step"] = _pick(NEXT_STEP[levels["next_step"]], rng, implicit_share)

        turns = [("customer", scenario.opening_line)]
        turns.append(
            (
                "trainee",
                f"Thanks for making the time. {questions[0] if questions else 'I wanted to walk you through the renewal.'}",
            )
        )
        turns.append(("customer", rng.choice(scenario.reactions)))
        if len(questions) > 1:
            turns.append(("trainee", questions[1]))
            turns.append(("customer", rng.choice(scenario.reactions)))
        turns.append(("customer", scenario.objections[i % len(scenario.objections)]))
        turns.append(("trainee", f"{empathy} {objection}"))
        turns.append(("customer", rng.choice(scenario.reactions)))
        turns.append(("trainee", value))
        turns.append(("customer", rng.choice(scenario.reactions)))
        turns.append(("trainee", step))
        turns.append(("customer", "Okay. Talk then."))
        rows.append(
            {
                "id": f"t{i:03d}",
                "scenario_id": scenario.id,
                "turns": [{"role": r, "content": c} for r, c in turns],
                "labels": levels,
                "phrasing": kinds,
            }
        )
    return rows


def main() -> None:
    rows = build()
    path = Path(__file__).parent / "data" / "transcripts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} transcripts to {path}")


if __name__ == "__main__":
    main()
