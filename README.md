# KidSpark

A Python CLI that generates three age-appropriate activities for a child,
given their age, interests, available time, materials on hand, and location.
Uses OpenAI's structured-output API so responses are always valid and typed.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then edit .env and paste your real key
```

## Usage

```bash
python main.py \
  --age 6 \
  --interests dinosaurs drawing \
  --minutes 30 \
  --materials paper crayons \
  --location indoors
```

Add `--verbose` to see the generator's debug logs on stderr.

### Flags

| Flag           | Required | Description                                             |
|----------------|----------|---------------------------------------------------------|
| `--age`        | yes      | Child's age in years (2-14).                            |
| `--interests`  | yes      | One or more things the child enjoys, space-separated.   |
| `--minutes`    | yes      | Time budget per activity (5-240).                       |
| `--materials`  | no       | Materials on hand. Omit if nothing is available.        |
| `--location`   | yes      | Where it will happen, e.g. `indoors`, `backyard`.       |
| `--verbose`    | no       | Show debug logs on stderr.                              |

## Running tests

```bash
pytest tests/
```

Tests use a mocked OpenAI client, so they run offline and cost nothing.

## Project layout

```
kidspark/
├── models.py              Pydantic schemas for request and response
├── tools.py               LLM-callable tools (milestones, safety) and schemas
├── generator.py           Two-phase ReAct agent, system prompt, error handling
├── main.py                argparse CLI, result formatting
├── tests/test_generator.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Architecture

KidSpark is a two-phase ReAct agent:

**Phase 1 — Tool-calling loop** (`chat.completions.create`)
- Agent calls `get_developmental_milestones(age)` to ground design in real data
- Agent generates 3 candidate activities
- Agent calls `check_safety(description, age)` for each candidate
- If safety concerns are flagged, the agent rewrites activities and re-checks
- Loops up to 8 iterations or until LLM has no more tool calls

**Phase 2 — Structured output** (`chat.completions.parse`)
- Conversation history is collapsed into a final structured response
- `response_format=ActivityResponse` guarantees a validated Pydantic object
- Returns 3 activities with title, duration, materials, learning goals, safety notes

## Why two phases?
OpenAI's structured output and tool calling don't combine cleanly. The two-phase split lets the LLM reason flexibly with tools, then lock the final answer to a Pydantic schema.
