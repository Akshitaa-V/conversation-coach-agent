from __future__ import annotations

from collections import Counter

from coach.experiments import assign
from coach.llm.base import estimate_tokens
from coach.memory import ConversationMemory
from coach.retrieval import BM25
from coach.store import SessionStore


def test_memory_stays_within_budget_and_summarises_old_turns():
    memory = ConversationMemory(token_budget=200)
    for i in range(30):
        memory.add("user", f"Trainee point number {i}. " + "detail " * 20)
        memory.add("assistant", f"Customer answer {i}. " + "reply " * 20)
    messages = memory.context("[task:customer] system prompt")
    total = sum(estimate_tokens(m["content"]) + 4 for m in messages)
    assert total <= 200
    assert memory.summarised_turns > 0
    assert "Earlier in this call" in messages[0]["content"]
    assert messages[-1]["content"].startswith("Customer answer 29")


def test_memory_keeps_everything_when_it_fits():
    memory = ConversationMemory(token_budget=2000)
    memory.add("user", "Hi")
    memory.add("assistant", "Hello")
    messages = memory.context("system")
    assert [m["content"] for m in messages[1:]] == ["Hi", "Hello"]
    assert memory.summarised_turns == 0


def test_long_term_memory_returns_weakest_dimensions(tmp_path):
    store = SessionStore(str(tmp_path / "s.db"))
    store.save_focus(
        "t1", {"discovery": 2, "empathy": 5, "value": 1, "objection_handling": 3, "next_step": 4}
    )
    assert store.weakest_dimensions("t1") == ["value", "discovery"]
    store.save_focus("t1", {"value": 5})
    assert store.weakest_dimensions("t1") == ["discovery", "objection_handling"]


def test_assignment_is_sticky_and_balanced():
    assert assign("exp", "trainee-1") == assign("exp", "trainee-1")
    counts = Counter(assign("exp", f"t{i}") for i in range(10_000))
    assert abs(counts["control"] / 10_000 - 0.5) < 0.02


def test_assignment_differs_between_experiments():
    same = sum(assign("exp-a", f"t{i}") == assign("exp-b", f"t{i}") for i in range(2000))
    assert 800 < same < 1200  # independent experiments are not correlated


def test_bm25_ranks_relevant_guidance_first():
    top = BM25().search("customer feels frustrated on hold", k=1)[0]
    assert top.dimension == "empathy"
