"""Tests for the generator module.

These tests never hit the network. We build a fake OpenAI client whose
`beta.chat.completions.parse` method returns (or raises) whatever the test
needs, and pass it in via the `client` parameter on generate_activities.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openai import OpenAIError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator import generate_activities  # noqa: E402
from models import Activity, ActivityRequest, ActivityResponse  # noqa: E402


def _sample_request() -> ActivityRequest:
    """Build a valid request reused across tests."""
    return ActivityRequest(
        age=6,
        interests=["dinosaurs", "drawing"],
        available_minutes=30,
        materials=["paper", "crayons"],
        location="indoors",
    )


def _sample_response() -> ActivityResponse:
    """Build a valid response the fake client will 'return'."""
    activity = Activity(
        title="Dino Fossil Dig",
        description="Hide paper dino cutouts and let the child uncover them.",
        estimated_minutes=20,
        materials_used=["paper", "crayons"],
        learning_goals=["observation", "fine motor skills"],
        safety_notes=["Supervise use of scissors if cutting paper."],
    )
    return ActivityResponse(activities=[activity, activity, activity])


def _fake_client_returning(response: ActivityResponse) -> MagicMock:
    """Build a MagicMock shaped like the OpenAI client's parse call chain."""
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.parsed = response
    completion.choices[0].message.refusal = None
    client.beta.chat.completions.parse.return_value = completion
    return client


def test_generate_activities_returns_parsed_response() -> None:
    """Happy path: the generator returns whatever the client parsed."""
    expected = _sample_response()
    client = _fake_client_returning(expected)

    result = generate_activities(_sample_request(), client=client)

    assert result is expected
    assert len(result.activities) == 3
    client.beta.chat.completions.parse.assert_called_once()
    call_kwargs = client.beta.chat.completions.parse.call_args.kwargs
    assert call_kwargs["response_format"] is ActivityResponse
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"
    assert "age: 6" in call_kwargs["messages"][1]["content"]


def test_generate_activities_wraps_openai_errors_as_runtime_error() -> None:
    """SDK errors should surface as RuntimeError for the CLI to catch."""
    client = MagicMock()
    client.beta.chat.completions.parse.side_effect = OpenAIError("boom")

    with pytest.raises(RuntimeError, match="OpenAI API call failed"):
        generate_activities(_sample_request(), client=client)
