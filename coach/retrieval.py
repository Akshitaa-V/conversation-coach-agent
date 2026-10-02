"""Small BM25 index over coaching guidance, used by the coach's ``find_examples`` tool."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass(frozen=True)
class Snippet:
    id: str
    dimension: str
    text: str


GUIDANCE: tuple[Snippet, ...] = (
    Snippet(
        "d1",
        "discovery",
        "Open with a question about the customer's current setup before you mention any product.",
    ),
    Snippet(
        "d2",
        "discovery",
        "Ask what success would look like in six months; it gives you the language for your value statement.",
    ),
    Snippet(
        "d3",
        "discovery",
        "Follow a short answer with 'what makes that important right now?' to find the real driver.",
    ),
    Snippet(
        "e1",
        "empathy",
        "Name the feeling in the customer's words: 'waiting 40 minutes on hold is frustrating' beats 'I understand'.",
    ),
    Snippet(
        "e2",
        "empathy",
        "Acknowledge first, solve second. An answer given before the customer feels heard sounds like a defence.",
    ),
    Snippet(
        "e3",
        "empathy",
        "Apologise for the specific mistake, not in general, and say what you will do about it.",
    ),
    Snippet(
        "v1",
        "value",
        "Translate features into the customer's numbers: hours saved per week, outages avoided, euros per month.",
    ),
    Snippet(
        "v2",
        "value",
        "Reuse what the customer told you in discovery when you state the benefit, so it is clearly about them.",
    ),
    Snippet(
        "v3",
        "value",
        "One concrete, quantified benefit is more convincing than a list of five features.",
    ),
    Snippet(
        "o1",
        "objection_handling",
        "On a price objection, ask 'compared to what?' before defending the price.",
    ),
    Snippet(
        "o2",
        "objection_handling",
        "Separate the objection from the person: 'is it the price itself, or the budget timing?'",
    ),
    Snippet(
        "o3",
        "objection_handling",
        "Do not argue with a competitor quote; ask what is included and reframe on total cost and risk.",
    ),
    Snippet(
        "o4",
        "objection_handling",
        "After you answer an objection, check it landed: 'does that address your concern about the outages?'",
    ),
    Snippet(
        "n1",
        "next_step",
        "End with a specific action, an owner and a date: 'I will send the comparison by Thursday 10 am'.",
    ),
    Snippet(
        "n2",
        "next_step",
        "Propose the next step yourself and ask for agreement; 'let's stay in touch' is not a next step.",
    ),
    Snippet(
        "n3",
        "next_step",
        "Send a calendar invite while you are still on the call when the customer agrees to a follow-up.",
    ),
)


class BM25:
    def __init__(
        self, snippets: tuple[Snippet, ...] = GUIDANCE, k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.snippets = snippets
        self.k1, self.b = k1, b
        self.docs = [tokenize(s.text) for s in snippets]
        self.avg_len = sum(len(d) for d in self.docs) / len(self.docs)
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3, dimension: str | None = None) -> list[Snippet]:
        terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self.docs):
            if dimension and self.snippets[i].dimension != dimension:
                continue
            tf = Counter(doc)
            score = 0.0
            for term in terms:
                if term not in tf:
                    continue
                num = tf[term] * (self.k1 + 1)
                den = tf[term] + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_len)
                score += self.idf[term] * num / den
            scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.snippets[i] for _, i in scored[:k]]
