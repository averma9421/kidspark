"""LLM-callable tools for KidSpark.

Exposes two tools the model can invoke during activity generation:
- get_developmental_milestones: age-band lookup of physical/cognitive/social
  benchmarks plus attention span and supervision level.
- check_safety: keyword + age scan of an activity description that flags
  age-inappropriate hazards.

Also exports OpenAI-style function-calling schemas (TOOL_SCHEMAS) and a
name->callable dispatcher (TOOL_REGISTRY) so the generator can hand both
to the OpenAI client and route tool calls back to Python.
"""

from typing import Callable, Dict, List, Tuple


_MILESTONES: List[Tuple[Tuple[int, int], Dict]] = [
    (
        (2, 3),
        {
            "physical_skills": [
                "walks and runs with growing confidence",
                "climbs on low furniture",
                "stacks 4-6 blocks",
                "scribbles with crayons",
                "kicks and throws a ball with limited aim",
            ],
            "cognitive": [
                "follows simple one-step instructions",
                "names familiar objects and body parts",
                "begins simple pretend play",
                "sorts by shape and primary color",
                "vocabulary expands rapidly",
            ],
            "social": [
                "engages in parallel play alongside peers",
                "imitates adults and older children",
                "asserts independence ('me do it')",
                "expresses emotions intensely; limited self-regulation",
            ],
            "attention_span_minutes": 5,
            "supervision_level": "constant",
        },
    ),
    (
        (4, 6),
        {
            "physical_skills": [
                "hops on one foot and skips",
                "uses safety scissors with guidance",
                "draws recognizable shapes and people",
                "throws and catches a medium ball",
                "buttons clothes and begins tying laces",
            ],
            "cognitive": [
                "follows two- and three-step instructions",
                "counts to 20 and recognizes letters",
                "asks frequent 'why' and 'how' questions",
                "retells parts of stories and simple sequences",
                "begins to grasp time concepts (today, later)",
            ],
            "social": [
                "engages in cooperative and role-based play",
                "takes turns with reminders",
                "forms early friendships and shows empathy",
                "tests rules but responds to clear limits",
            ],
            "attention_span_minutes": 15,
            "supervision_level": "active",
        },
    ),
    (
        (7, 9),
        {
            "physical_skills": [
                "rides a bike without training wheels",
                "writes legibly and uses basic tools",
                "improved hand-eye coordination for sports",
                "builds with detailed construction sets",
                "endurance for sustained physical play",
            ],
            "cognitive": [
                "reads independently and writes short pieces",
                "applies logic to concrete problems",
                "understands money, time, and basic measurement",
                "follows multi-step instructions reliably",
                "begins planning simple projects",
            ],
            "social": [
                "forms close, often same-gender friendships",
                "values fairness and rules strongly",
                "compares own abilities to peers",
                "works productively in small groups",
            ],
            "attention_span_minutes": 25,
            "supervision_level": "moderate",
        },
    ),
    (
        (10, 14),
        {
            "physical_skills": [
                "fine motor mastery for crafts and instruments",
                "coordinated, complex sports skills",
                "puberty-driven changes in size and strength",
                "stamina for extended focused activity",
            ],
            "cognitive": [
                "abstract and hypothetical thinking emerges",
                "plans, organizes, and executes multi-step projects",
                "researches topics independently",
                "evaluates evidence and forms opinions",
                "reflects on own thinking (metacognition)",
            ],
            "social": [
                "peer group becomes a primary reference point",
                "tests boundaries with adults; values autonomy",
                "develops a clearer personal identity",
                "navigates conflict and negotiation more independently",
            ],
            "attention_span_minutes": 45,
            "supervision_level": "minimal",
        },
    ),
]


_SAFETY_RULES: Dict[str, List[Tuple[int, str]]] = {
    "scissors": [
        (5, "Scissors at this age require constant supervision"),
        (8, "Use safety scissors and supervise cutting closely"),
    ],
    "knife": [
        (8, "Knives are not appropriate without direct adult control"),
        (12, "Use only kid-safe knives with active adult supervision"),
    ],
    "hot": [
        (4, "Hot items pose serious burn risk; keep the child well clear"),
        (9, "Active supervision required around any heat source"),
    ],
    "small parts": [
        (4, "Small parts: choking hazard for this age"),
        (6, "Verify no parts are small enough to be swallowed"),
    ],
    "battery": [
        (6, "Button or coin batteries are a serious ingestion hazard; keep sealed"),
        (12, "Ensure battery compartments are screwed shut before play"),
    ],
    "rope": [
        (6, "Rope poses strangulation risk; never near the neck and supervise length"),
        (10, "Limit rope length and avoid loops around limbs or neck"),
    ],
    "string": [
        (4, "String or yarn is a strangulation and ingestion hazard at this age"),
        (8, "Keep string lengths short and supervise use"),
    ],
    "water": [
        (3, "Water activities require active supervision"),
        (7, "Active supervision near water; even shallow water is a drowning risk"),
    ],
    "stairs": [
        (3, "Stairs require gating or hand-holding at this age"),
        (6, "Supervise on stairs, especially when carrying objects"),
    ],
}


def get_developmental_milestones(age: int) -> Dict:
    """Return age-appropriate developmental benchmarks for a child.

    Looks up a hardcoded age band (2-3, 4-6, 7-9, 10-14) and returns physical,
    cognitive, and social skill lists alongside a typical attention span in
    minutes and a recommended supervision level.

    Args:
        age: Child's age in whole years. Ages outside 2-14 are clamped to the
            nearest supported band so the tool always returns usable guidance.

    Returns:
        Dict with keys: physical_skills (List[str]), cognitive (List[str]),
        social (List[str]), attention_span_minutes (int), and
        supervision_level (str).
    """
    clamped = max(2, min(14, age))
    for (low, high), data in _MILESTONES:
        if low <= clamped <= high:
            return dict(data)
    return dict(_MILESTONES[-1][1])


def check_safety(activity_description: str, age: int) -> List[str]:
    """Scan an activity description for age-inappropriate hazards.

    Performs case-insensitive substring matching against a fixed keyword list.
    Each keyword has tiered age thresholds; the first threshold the child's
    age is at or below produces the matching concern. A child older than every
    threshold for a keyword triggers no concern for it.

    Args:
        activity_description: Free-text description of the planned activity.
        age: Child's age in whole years.

    Returns:
        A list of concern strings. Empty list means no concerns were detected.
    """
    text = activity_description.lower()
    concerns: List[str] = []
    for keyword, tiers in _SAFETY_RULES.items():
        if keyword not in text:
            continue
        for max_age, message in tiers:
            if age <= max_age:
                concerns.append(message)
                break
    return concerns


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_developmental_milestones",
            "description": (
                "Look up age-appropriate developmental milestones for a child "
                "aged 2-14. Returns physical, cognitive, and social skill "
                "expectations plus a typical attention span and recommended "
                "supervision level. Call this before designing activities to "
                "ground your suggestions in realistic capabilities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 14,
                        "description": "Child's age in whole years (2-14).",
                    }
                },
                "required": ["age"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_safety",
            "description": (
                "Check a proposed activity description for age-inappropriate "
                "safety concerns. Performs keyword matching for hazards like "
                "scissors, knives, hot items, small parts, batteries, rope, "
                "string, water, and stairs, and returns concerns scaled to "
                "the child's age. An empty list means no flagged hazards."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "activity_description": {
                        "type": "string",
                        "description": (
                            "Plain-text description of the activity, including "
                            "materials and steps."
                        ),
                    },
                    "age": {
                        "type": "integer",
                        "minimum": 2,
                        "maximum": 14,
                        "description": "Child's age in whole years (2-14).",
                    },
                },
                "required": ["activity_description", "age"],
                "additionalProperties": False,
            },
        },
    },
]


TOOL_REGISTRY: Dict[str, Callable] = {
    "get_developmental_milestones": get_developmental_milestones,
    "check_safety": check_safety,
}
