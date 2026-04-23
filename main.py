"""KidSpark CLI entry point.

Parses command-line arguments, builds an ActivityRequest, invokes the
generator, and prints a human-readable summary of the three activities.
"""

import argparse
import sys
from typing import List

from loguru import logger
from pydantic import ValidationError

from generator import generate_activities
from models import Activity, ActivityRequest, ActivityResponse


def _parse_args(argv: List[str]) -> argparse.Namespace:
    """Define and parse the CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="kidspark",
        description="Generate three age-appropriate activities for a child.",
    )
    parser.add_argument(
        "--age",
        type=int,
        required=True,
        help="Child's age in years (2-14).",
    )
    parser.add_argument(
        "--interests",
        nargs="+",
        required=True,
        metavar="INTEREST",
        help="Things the child enjoys, space-separated. e.g. --interests dinosaurs drawing",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        required=True,
        help="Time budget in minutes (5-240).",
    )
    parser.add_argument(
        "--materials",
        nargs="*",
        default=[],
        metavar="MATERIAL",
        help="Materials on hand, space-separated. Omit if nothing is available.",
    )
    parser.add_argument(
        "--location",
        required=True,
        help="Where the activity will happen, e.g. 'indoors', 'backyard'.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show debug logs from the generator.",
    )
    return parser.parse_args(argv)


def _configure_logging(verbose: bool) -> None:
    """Route loguru to stderr so it never mixes with the printed results."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")


def _format_activity(index: int, activity: Activity) -> str:
    """Render a single activity as a readable block for the terminal."""
    lines = [
        f"Activity {index}: {activity.title}",
        f"  Time: ~{activity.estimated_minutes} min",
        f"  Materials: {', '.join(activity.materials_used) or 'none'}",
        "",
        "  Description:",
        f"    {activity.description}",
        "",
        "  Learning goals:",
        *[f"    - {goal}" for goal in activity.learning_goals],
        "",
        "  Safety notes:",
        *[f"    - {note}" for note in activity.safety_notes],
    ]
    return "\n".join(lines)


def _print_response(response: ActivityResponse) -> None:
    """Print all activities separated by a visible divider."""
    divider = "\n" + ("-" * 60) + "\n"
    blocks = [
        _format_activity(i, activity)
        for i, activity in enumerate(response.activities, start=1)
    ]
    print(divider.join(blocks))


def main(argv: List[str]) -> int:
    """Run the CLI. Returns a process exit code (0 success, 1 failure)."""
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    try:
        request = ActivityRequest(
            age=args.age,
            interests=args.interests,
            available_minutes=args.minutes,
            materials=args.materials,
            location=args.location,
        )
    except ValidationError as e:
        print(f"Invalid input: {e}", file=sys.stderr)
        return 1

    try:
        response = generate_activities(request)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    _print_response(response)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
