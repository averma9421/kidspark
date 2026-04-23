"""Core LLM integration for KidSpark.

Takes a validated ActivityRequest, builds a structured prompt, calls OpenAI
with a Pydantic response schema, and returns a validated ActivityResponse.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI, OpenAIError

from models import ActivityRequest, ActivityResponse


MODEL_NAME = "gpt-4o-mini"

SYSTEM_PROMPT = """\
# Role
You are KidSpark, an expert in child development and creative play. You design
safe, age-appropriate, engaging activities for children aged 2-14.

# Context
A caregiver will give you a child's age, interests, time budget, available
materials, and location. You must propose activities the caregiver can run
today with what they already have on hand.

# Task
Propose exactly three distinct activities tailored to the child. Each activity
must include a title, a step-by-step description, an estimated duration,
the subset of materials it uses, learning goals, and safety notes.

# Constraints
- Respect the child's developmental stage for the given age. Vocabulary,
  dexterity expectations, and attention span must match the age.
- Use ONLY materials from the provided list. Do not introduce new materials.
  If the list is empty, propose activities that need no materials.
- Each activity's estimated duration must fit within the caregiver's time
  budget. The total is per-activity, not cumulative.
- The activity must be appropriate for the stated location (e.g. no running
  games indoors in a small apartment).
- Tie each activity to at least one of the child's interests.
- Always include at least one concrete safety note per activity, written for
  the caregiver, not the child.
- The three activities should be meaningfully different from each other
  (different skills, formats, or energy levels), not variations of one idea.
"""


def _build_user_message(request: ActivityRequest) -> str:
    """Render the validated request into a plain-text user message.

    Keeping this separate from the system prompt makes the call easy to test
    and keeps per-request data out of the cached system prompt.
    """
    materials = ", ".join(request.materials) if request.materials else "none"
    interests = ", ".join(request.interests)
    return (
        f"Child's age: {request.age}\n"
        f"Interests: {interests}\n"
        f"Available time: {request.available_minutes} minutes\n"
        f"Materials on hand: {materials}\n"
        f"Location: {request.location}"
    )


def _get_client(api_key: Optional[str] = None) -> OpenAI:
    """Build an OpenAI client, loading the API key from .env if not provided.

    Raises RuntimeError with a clear message if no key is found, so the CLI
    can surface a helpful error instead of an opaque SDK failure.
    """
    load_dotenv()
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=key)


def generate_activities(
    request: ActivityRequest,
    client: Optional[OpenAI] = None,
) -> ActivityResponse:
    """Generate three age-appropriate activities for the given request.

    Args:
        request: Validated user input describing the child and constraints.
        client: Optional pre-built OpenAI client. Injected in tests; in normal
            use we build one from the environment.

    Returns:
        An ActivityResponse containing exactly three Activity objects.

    Raises:
        RuntimeError: If the API key is missing, the API call fails, or the
            model returns an empty/unparseable response.
    """
    client = client or _get_client()
    user_message = _build_user_message(request)
    logger.info(
        "Requesting activities: age={} interests={} minutes={}",
        request.age,
        request.interests,
        request.available_minutes,
    )

    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=ActivityResponse,
        )
    except OpenAIError as e:
        logger.error("OpenAI API call failed: {}", e)
        raise RuntimeError(f"OpenAI API call failed: {e}") from e

    if not completion.choices:
        raise RuntimeError("OpenAI returned no choices.")

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        refusal = completion.choices[0].message.refusal
        if refusal:
            raise RuntimeError(f"Model refused to respond: {refusal}")
        raise RuntimeError("Model returned an empty response.")

    logger.info("Received {} activities", len(parsed.activities))
    return parsed
