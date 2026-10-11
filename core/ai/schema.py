"""The shape the AI's answer must have.

Free text can't be checked. So the AI must return JSON where every sentence
comes with the ids of the facts it rests on. Anything outside this shape
(extra keys, missing citations, too many items) is rejected before any
guardrail runs.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Bullet(_Strict):
    text: str = Field(min_length=10, max_length=450)
    facts: list[str] = Field(min_length=1, max_length=6)  # ids from the facts sheet


class Recommendation(Bullet):
    type: str = Field(min_length=1, max_length=40)  # must be in the use case's allowed list


class InsightDoc(_Strict):
    executive_summary: list[Bullet] = Field(min_length=2, max_length=6)
    key_findings: list[Bullet] = Field(min_length=1, max_length=6)
    recommendations: list[Recommendation] = Field(max_length=5)
    caveats: list[Bullet] = Field(min_length=1, max_length=5)


def json_schema() -> dict:
    return InsightDoc.model_json_schema()
