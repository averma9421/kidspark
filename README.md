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
├── generator.py           OpenAI call, system prompt, error handling
├── main.py                argparse CLI, result formatting
├── tests/test_generator.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## How it works

`main.py` parses CLI flags into an `ActivityRequest`. `generator.py` builds
a user message from the request, sends it to `gpt-4o-mini` alongside a system
prompt that defines the role, context, task, and constraints, and asks the
API for output matching the `ActivityResponse` Pydantic schema. The parsed
response is printed as three readable activity blocks.
