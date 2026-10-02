"""Typed contracts shared by the API, the agents and the evals."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Dimension = Literal["discovery", "empathy", "value", "objection_handling", "next_step"]
DIMENSIONS: tuple[str, ...] = ("discovery", "empathy", "value", "objection_handling", "next_step")


class DimensionScore(BaseModel):
    dimension: Dimension
    score: int = Field(ge=1, le=5)
    evidence: str = Field(
        default="",
        max_length=300,
        description="Verbatim quote from a trainee turn that supports the score, or empty.",
    )


class Feedback(BaseModel):
    """Structured output of the coach agent. Validated before it reaches a user."""

    scores: list[DimensionScore]
    strengths: list[str] = Field(default_factory=list, max_length=3)
    next_focus: str = Field(min_length=3, max_length=300)

    @field_validator("scores")
    @classmethod
    def one_score_per_dimension(cls, scores: list[DimensionScore]) -> list[DimensionScore]:
        seen = [s.dimension for s in scores]
        if sorted(seen) != sorted(DIMENSIONS):
            raise ValueError(f"need exactly one score for each of {DIMENSIONS}, got {seen}")
        return scores

    @property
    def overall(self) -> float:
        return round(sum(s.score for s in self.scores) / len(self.scores), 2)

    def score_of(self, dimension: str) -> int:
        return next(s.score for s in self.scores if s.dimension == dimension)


class Turn(BaseModel):
    role: Literal["trainee", "customer"]
    content: str


class CreateSessionRequest(BaseModel):
    trainee_id: str = Field(min_length=1, max_length=64)
    scenario_id: str


class SessionInfo(BaseModel):
    session_id: str
    scenario_id: str
    variant: str
    opening_line: str


class JudgeScores(BaseModel):
    """Output of an LLM judge scoring a finished transcript against the rubric."""

    discovery: int = Field(ge=1, le=5)
    empathy: int = Field(ge=1, le=5)
    value: int = Field(ge=1, le=5)
    objection_handling: int = Field(ge=1, le=5)
    next_step: int = Field(ge=1, le=5)

    def as_dict(self) -> dict[str, int]:
        return {d: getattr(self, d) for d in DIMENSIONS}


class PairwiseVerdict(BaseModel):
    winner: Literal["A", "B", "tie"]

    @model_validator(mode="before")
    @classmethod
    def normalise(cls, data):
        if isinstance(data, dict) and isinstance(data.get("winner"), str):
            data = {**data, "winner": data["winner"].strip().upper().replace("TIE", "tie")}
        return data
