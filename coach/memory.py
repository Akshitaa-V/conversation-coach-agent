"""Context engineering for long roleplays, plus long-term memory per trainee.

Short-term: the customer agent sees the system prompt, a running summary of
older turns and as many recent turns as fit a token budget. Older turns are
folded into the summary instead of being silently cut, so the customer does not
"forget" an objection it raised ten turns ago.

Long-term: after each feedback, the trainee's weakest dimensions are stored
(see ``store.SessionStore.save_focus``) and handed to the coach next time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from coach.llm.base import Message, estimate_tokens


@dataclass
class ConversationMemory:
    token_budget: int = 1200
    turns: list[Message] = field(default_factory=list)
    summary_lines: list[str] = field(default_factory=list)
    _summarised_upto: int = 0

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})

    @staticmethod
    def _one_line(message: Message) -> str:
        speaker = "Trainee" if message["role"] == "user" else "Customer"
        text = message["content"].strip().replace("\n", " ")
        first_sentence = text.split(". ")[0][:160]
        return f"{speaker}: {first_sentence}"

    def context(self, system_prompt: str) -> list[Message]:
        """Build the message list for the next model call within ``token_budget``."""
        budget = self.token_budget - estimate_tokens(system_prompt) - 8
        recent: list[Message] = []
        used = 0
        for message in reversed(self.turns[self._summarised_upto :]):
            cost = estimate_tokens(message["content"]) + 4
            if used + cost > budget * 0.75 and recent:
                break
            recent.insert(0, message)
            used += cost

        first_kept = len(self.turns) - len(recent)
        for message in self.turns[self._summarised_upto : first_kept]:
            self.summary_lines.append(self._one_line(message))
        self._summarised_upto = max(self._summarised_upto, first_kept)

        # Keep the summary inside the remaining quarter of the budget, oldest lines first out.
        while self.summary_lines and estimate_tokens("\n".join(self.summary_lines)) > budget * 0.25:
            self.summary_lines.pop(0)

        system = system_prompt
        if self.summary_lines:
            system += "\n\nEarlier in this call (summary):\n" + "\n".join(self.summary_lines)
        return [{"role": "system", "content": system}, *recent]

    @property
    def summarised_turns(self) -> int:
        return self._summarised_upto
