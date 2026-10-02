"""Customer persona agent: plays the customer in real time and streams its reply."""

from __future__ import annotations

from collections.abc import AsyncIterator

from coach.experiments import VARIANTS
from coach.llm.router import ModelRouter, StreamResult
from coach.memory import ConversationMemory
from coach.scenarios import Scenario

FALLBACK_LINES = (
    "Sorry, could you say that again? I got distracted for a second.",
    "Hold on, someone just walked in. Can you repeat the last part?",
)


def system_prompt(scenario: Scenario, variant: str) -> str:
    v = VARIANTS[variant]
    objections = "\n".join(f"- {o}" for o in scenario.objections)
    return (
        "[task:customer]\n"
        f"You are role-playing a customer in a sales and service training call (scenario: {scenario.id}) "
        f"(variant: {v.name}).\n"
        f"Who you are: {scenario.persona}\n"
        f"Situation: {scenario.situation}\n"
        "Objections you hold, to raise naturally when the conversation gives you a reason:\n"
        f"{objections}\n"
        "Rules: stay in character, never coach the trainee, never mention that this is a simulation. "
        "Soften when the trainee listens and gives specifics; push back when they pitch without listening. "
        f"{v.customer_instructions}"
    )


class CustomerAgent:
    def __init__(
        self, router: ModelRouter, scenario: Scenario, variant: str, memory: ConversationMemory
    ) -> None:
        self.router = router
        self.scenario = scenario
        self.variant = variant
        self.memory = memory
        self._system = system_prompt(scenario, variant)

    async def reply(
        self, trainee_message: str, result: StreamResult, conversation_id: str
    ) -> AsyncIterator[str]:
        self.memory.add("user", trainee_message)
        messages = self.memory.context(self._system)
        parts: list[str] = []
        async for delta in self.router.stream(
            "customer",
            messages,
            result,
            conversation_id=conversation_id,
            temperature=VARIANTS[self.variant].temperature,
        ):
            parts.append(delta)
            yield delta
        self.memory.add("assistant", "".join(parts))

    def remember_reply(self, text: str) -> None:
        """Store a reply that did not come from a clean stream (partial or canned)."""
        self.memory.add("assistant", text)
