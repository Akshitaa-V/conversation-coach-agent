"""Scenario library and the shared five-dimension rubric."""

from __future__ import annotations

from dataclasses import dataclass

RUBRIC: dict[str, str] = {
    "discovery": (
        "Asks open questions about the customer's situation, goals and constraints before "
        "pitching. 1 = no questions, 3 = one relevant question, 5 = several open questions "
        "that shape the rest of the call."
    ),
    "empathy": (
        "Acknowledges the customer's concern or feeling in their own terms before answering. "
        "1 = ignores it, 3 = generic acknowledgement, 5 = specific, sincere acknowledgement."
    ),
    "value": (
        "Links the offer to an outcome the customer cares about. 1 = features only, "
        "3 = general benefit, 5 = a concrete, quantified benefit tied to what the customer said."
    ),
    "objection_handling": (
        "Handles the objection by exploring it and reframing, not by dismissing or ignoring it. "
        "1 = ignored, 2 = dismissed or argued, 4-5 = explored and reframed with evidence."
    ),
    "next_step": (
        "Closes with a clear, agreed next step. 1 = none, 3 = vague ('let's stay in touch'), "
        "5 = specific action with an owner and a date or time."
    ),
}


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    persona: str
    situation: str
    objections: tuple[str, ...]
    opening_line: str
    reactions: tuple[str, ...]


SCENARIOS: dict[str, Scenario] = {
    s.id: s
    for s in (
        Scenario(
            id="renewal_price_increase",
            title="B2B contract renewal with a price increase",
            persona=(
                "Head of IT at a 400-person logistics company. Busy, data-driven, mildly "
                "annoyed about a 12% price increase on the connectivity contract."
            ),
            situation="The contract renews in six weeks. A competitor sent a cheaper offer.",
            objections=(
                "Honestly, a 12% increase is hard to justify to my CFO.",
                "Your competitor quoted us almost 15% less for the same bandwidth.",
                "We had two outages last quarter, so I am not sure the premium is worth it.",
            ),
            opening_line="Hi, thanks for calling. I only have about 15 minutes, so let's get to the renewal.",
            reactions=(
                "Okay, that is fair.",
                "Go on.",
                "I hear you, but my budget is fixed this year.",
                "That would help, actually.",
            ),
        ),
        Scenario(
            id="fleet_discovery_call",
            title="First discovery call with a fleet operations manager",
            persona=(
                "Fleet operations manager for 180 delivery trucks. Practical, sceptical of "
                "software vendors, cares about downtime and driver turnover."
            ),
            situation="Inbound lead after a webinar. No budget allocated yet.",
            objections=(
                "We already tried a telematics tool and the drivers hated it.",
                "I don't have budget for this until next fiscal year.",
                "My team has no time to learn another system.",
            ),
            opening_line="Hi. I signed up after the webinar, but to be honest I am just looking around.",
            reactions=(
                "Hm, maybe.",
                "That is a real problem for us, yes.",
                "I would need to see numbers.",
                "Okay.",
            ),
        ),
        Scenario(
            id="billing_complaint",
            title="Service call: angry customer about a double charge",
            persona=(
                "Long-time private customer, frustrated after being charged twice and waiting "
                "40 minutes on hold. Wants it fixed, not explained."
            ),
            situation="The duplicate charge is real; a refund takes 3-5 business days.",
            objections=(
                "Three to five days? That is ridiculous, it was your mistake.",
                "This is the second time this year. Why should I stay with you?",
                "I don't want a voucher, I want my money back.",
            ),
            opening_line="Finally. I have been charged twice this month and I have been on hold forever.",
            reactions=(
                "Fine.",
                "Okay, at least someone is listening.",
                "That is still not great.",
                "Alright, thank you.",
            ),
        ),
    )
}


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        raise KeyError(f"unknown scenario '{scenario_id}'") from exc
