"""Tools the coach agent can call, with JSON schemas in the OpenAI tool format."""

from __future__ import annotations

import json
from collections.abc import Callable

from coach.retrieval import BM25
from coach.scenarios import RUBRIC, SCENARIOS

_index = BM25()

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_rubric",
            "description": "Return the scenario description and the scoring rubric for all five dimensions.",
            "parameters": {
                "type": "object",
                "properties": {"scenario_id": {"type": "string", "enum": sorted(SCENARIOS)}},
                "required": ["scenario_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_examples",
            "description": "Search coaching guidance for concrete advice on one rubric dimension.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "enum": sorted(RUBRIC)},
                    "query": {"type": "string", "description": "What the trainee struggled with."},
                },
                "required": ["dimension", "query"],
            },
        },
    },
]


def get_rubric(scenario_id: str) -> dict:
    scenario = SCENARIOS[scenario_id]
    return {"scenario": scenario.title, "situation": scenario.situation, "rubric": RUBRIC}


def find_examples(dimension: str, query: str) -> dict:
    if dimension not in RUBRIC:
        raise ValueError(f"unknown dimension '{dimension}'")
    return {"examples": [s.text for s in _index.search(query, k=2, dimension=dimension)]}


TOOLS: dict[str, Callable[..., dict]] = {"get_rubric": get_rubric, "find_examples": find_examples}


def run_tool(name: str, arguments: str) -> str:
    """Execute a tool call. Errors are returned to the model as data so it can recover."""
    try:
        func = TOOLS[name]
        result = func(**json.loads(arguments or "{}"))
        return json.dumps(result)
    except KeyError:
        return json.dumps({"error": f"unknown tool or value in call to '{name}'"})
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return json.dumps({"error": f"bad arguments for '{name}': {exc}"})
