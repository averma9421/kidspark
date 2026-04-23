"""Pydantic data models for KidSpark.

These models define the shape of data flowing through the app:
- ActivityRequest: what the user asks for (input to the LLM call)
- Activity: one generated activity (structured LLM output piece)
- ActivityResponse: the full LLM response containing multiple Activities
"""

from typing import List
from pydantic import BaseModel, Field, field_validator


class ActivityRequest(BaseModel):
    """User input describing the child and constraints for activity generation."""

    age: int = Field(..., ge=2, le=14, description="Child's age in years (2-14).")
    interests: List[str] = Field(
        ...,
        min_length=1,
        description="Things the child enjoys, e.g. ['dinosaurs', 'drawing'].",
    )
    available_minutes: int = Field(
        ...,
        ge=5,
        le=240,
        description="Total time budget in minutes for one activity.",
    )
    materials: List[str] = Field(
        default_factory=list,
        description="Materials available at home, e.g. ['paper', 'crayons'].",
    )
    location: str = Field(
        ...,
        description="Where the activity will happen, e.g. 'indoors', 'backyard'.",
    )

    @field_validator("interests", "materials")
    @classmethod
    def _strip_and_drop_blanks(cls, v: List[str]) -> List[str]:
        """Trim whitespace and drop empty strings so the LLM sees clean input."""
        return [item.strip() for item in v if item and item.strip()]


class Activity(BaseModel):
    """A single concrete activity suggestion produced by the model."""

    title: str = Field(..., description="Short catchy name for the activity.")
    description: str = Field(
        ...,
        description="Step-by-step explanation a caregiver can follow.",
    )
    estimated_minutes: int = Field(
        ...,
        ge=1,
        description="How long the activity should take, in minutes.",
    )
    materials_used: List[str] = Field(
        ...,
        description="Subset of the provided materials this activity requires.",
    )
    learning_goals: List[str] = Field(
        ...,
        description="Developmental or educational benefits, e.g. 'fine motor skills'.",
    )
    safety_notes: List[str] = Field(
        ...,
        description="Age-appropriate safety considerations for a caregiver.",
    )


class ActivityResponse(BaseModel):
    """Top-level structured response returned by the LLM."""

    activities: List[Activity] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Exactly three activity suggestions.",
    )
