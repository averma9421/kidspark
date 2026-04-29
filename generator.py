"""Core LLM integration for KidSpark — Stage 2 ReAct agent.

Runs a two-phase pipeline:

Phase 1 — tool-calling loop. The LLM iterates with chat.completions.create,
optionally calling get_developmental_milestones and check_safety to ground
its reasoning in real developmental data and surface age-inappropriate
hazards before committing to a final design.

Phase 2 — structured output. Once the model has gathered what it needs we
ask it for the final answer via chat.completions.parse with the
ActivityResponse schema, guaranteeing a valid Pydantic object back.
"""

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI, OpenAIError

from models import ActivityRequest, ActivityResponse
from tools import TOOL_REGISTRY, TOOL_SCHEMAS


MODEL_NAME = "gpt-4o-mini"
MAX_ITERATIONS = 8

SYSTEM_PROMPT = """\
# Role
You are KidSpark, an expert in child development and creative play. You design
safe, age-appropriate, engaging activities for children aged 2-14.

# Context
A caregiver will give you a child's age, interests, time budget, available
materials, and location. You must propose three activities the caregiver can
run today with what they already have on hand.

# Available tools
- get_developmental_milestones(age): age-appropriate physical, cognitive, and
  social benchmarks plus typical attention span and supervision level.
- check_safety(activity_description, age): flags age-inappropriate hazards
  in a planned activity and returns a list of concerns (empty = clear).

# Required workflow
Step 1. Call get_developmental_milestones for the child's age first. Use the
        returned attention span and skill levels to ground your design.
Step 2. Draft three candidate activities that fit the milestones, the time
        budget, the materials, the location, and at least one interest.
Step 3. Call check_safety once per candidate, passing the activity's full
        description and the child's age.
Step 4. Refine any activity that came back with safety concerns — tighten
        wording, swap materials, or replace the activity entirely. Re-run
        check_safety on anything you change.
Step 5. When all three activities are well-grounded and clear of safety
        concerns you can resolve, stop calling tools and reply with a brief
        message indicating you are ready to produce the final structured
        response.

# Constraints
- Use ONLY materials from the provided list. If empty, propose activities
  that need no materials.
- Each activity's estimated duration must fit within the time budget.
- Always include at least one concrete safety note per activity, written
  for the caregiver, not the child.
- The three activities must be meaningfully different from each other
  (different skills, formats, or energy levels), not variations of one idea.
"""


def _build_user_message(request: ActivityRequest) -> str:
    """Render the validated request into a plain-text user message.

    Keeping this separate from the system prompt makes the call easy to test
    and keeps per-request data out of the cached system prompt.

    Args:
        request: Validated user input.

    Returns:
        A multi-line string describing the child and constraints.
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

    Args:
        api_key: Optional explicit key; if omitted we read OPENAI_API_KEY.

    Returns:
        A configured OpenAI client.
    """
    load_dotenv()
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=key)


def _truncate(text: str, limit: int = 200) -> str:
    """Return text shortened to `limit` characters with an ellipsis suffix.

    Args:
        text: String to shorten.
        limit: Maximum kept characters before the ellipsis.

    Returns:
        The original string if short enough, otherwise the prefix plus '...'.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _execute_tool_call(name: str, raw_args: str) -> str:
    """Dispatch a single model-issued tool call.

    Looks the function up in TOOL_REGISTRY, parses its JSON arguments, runs
    it, and returns a JSON-encoded result. Any failure (unknown name,
    invalid JSON, exception inside the tool) is converted to an
    "error: <reason>" string so the LLM can see what went wrong and adapt
    on the next iteration instead of the agent crashing.

    Args:
        name: Tool name as reported by the model.
        raw_args: JSON-encoded argument string from the tool_call.

    Returns:
        A string suitable for the "content" field of a tool message.
    """
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"error: unknown tool '{name}'"
    try:
        kwargs = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as e:
        return f"error: could not parse arguments: {e}"
    try:
        result = fn(**kwargs)
    except Exception as e:
        return f"error: tool {name} raised: {e}"
    return json.dumps(result)


def _run_tool_loop(
    client: OpenAI, messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Drive the Phase 1 ReAct loop and return the augmented message list.

    On each iteration we send the current history plus TOOL_SCHEMAS. If the
    model emits tool_calls we execute them, append the assistant turn and
    one tool message per call, and continue. If it emits no tool_calls we
    append the assistant turn and stop. We bail with RuntimeError if we
    exceed MAX_ITERATIONS without the model resolving.

    Args:
        client: OpenAI client (real or test double).
        messages: Conversation history; mutated in place and returned.

    Returns:
        The same list, with assistant and tool messages appended in order.

    Raises:
        RuntimeError: On API failure or if the loop hits MAX_ITERATIONS.
    """
    for iteration in range(1, MAX_ITERATIONS + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
        except OpenAIError as e:
            logger.error(
                "Phase 1 API call failed at iteration {}: {}", iteration, e
            )
            raise RuntimeError(
                f"OpenAI API call failed at iteration {iteration}: {e}"
            ) from e

        message = completion.choices[0].message
        tool_calls = message.tool_calls or []
        logger.info(
            "Phase 1 iter={} tool_calls={}", iteration, len(tool_calls)
        )

        if not tool_calls:
            messages.append(
                {"role": "assistant", "content": message.content or ""}
            )
            logger.info("Phase 1 complete after {} iteration(s)", iteration)
            return messages

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            args_preview = _truncate(tc.function.arguments or "")
            logger.info(
                "  tool_call name={} args={}",
                tc.function.name,
                args_preview,
            )
            content = _execute_tool_call(
                tc.function.name, tc.function.arguments
            )
            logger.info(
                "  tool_result name={} result={}",
                tc.function.name,
                _truncate(content),
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                }
            )

    raise RuntimeError("Agent did not finish within max iterations")


def generate_activities(
    request: ActivityRequest,
    client: Optional[OpenAI] = None,
) -> ActivityResponse:
    """Generate three age-appropriate activities via a two-phase agent.

    Phase 1 runs a tool-calling loop so the model can ground its design in
    developmental milestones and check each candidate for safety concerns.
    Phase 2 collapses the conversation into a structured ActivityResponse.

    Args:
        request: Validated user input describing the child and constraints.
        client: Optional pre-built OpenAI client. Injected in tests; in
            normal use we build one from the environment.

    Returns:
        An ActivityResponse containing exactly three Activity objects.

    Raises:
        RuntimeError: On missing API key, API failure, agent non-termination,
            or empty/refused model output.
    """
    client = client or _get_client()
    user_message = _build_user_message(request)
    logger.info(
        "Requesting activities: age={} interests={} minutes={}",
        request.age,
        request.interests,
        request.available_minutes,
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    messages = _run_tool_loop(client, messages)

    logger.info("Transitioning to Phase 2: structured output")
    messages.append(
        {
            "role": "user",
            "content": (
                "Based on your analysis above, now produce the final 3 "
                "activities as structured output."
            ),
        }
    )

    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=messages,
            response_format=ActivityResponse,
        )
    except OpenAIError as e:
        logger.error("Phase 2 API call failed: {}", e)
        raise RuntimeError(f"OpenAI API call failed in Phase 2: {e}") from e

    if not completion.choices:
        raise RuntimeError("OpenAI returned no choices.")

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        refusal = completion.choices[0].message.refusal
        if refusal:
            raise RuntimeError(f"Model refused to respond: {refusal}")
        raise RuntimeError("Model returned an empty response.")

    logger.info(
        "Phase 2 complete: returned {} activities", len(parsed.activities)
    )
    return parsed
