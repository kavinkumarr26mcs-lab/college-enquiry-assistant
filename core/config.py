"""Central configuration for the College Enquiry Assistant core package.

Nothing in this module imports Flask, so the values can be shared by the web
application, the evaluation script and the automated tests.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FAQ_PATH = DATA_DIR / "college_faqs.json"
DEFAULT_EVALUATION_PATH = DATA_DIR / "evaluation_questions.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

# Minimum adjusted similarity (0-1) required before an FAQ answer is returned.
# Override with the environment variable named below (see README).
DEFAULT_SIMILARITY_THRESHOLD = 0.35
THRESHOLD_ENV_VAR = "ENQUIRY_SIMILARITY_THRESHOLD"

# The raw cosine score is multiplied by (known_terms / all_terms) ** exponent so
# that a single shared word cannot carry an otherwise unrelated question.
# 0 disables this and gives plain cosine similarity.
DEFAULT_COVERAGE_EXPONENT = 0.5

# If the best FAQ beats the runner-up by less than this, the question is
# treated as ambiguous and the fallback is returned. 0 disables the check.
DEFAULT_MIN_MARGIN = 0.05

MAX_QUESTION_LENGTH = 500

FALLBACK_MESSAGE = (
    "I couldn't find a sufficiently reliable answer in the available FAQs. "
    "Please contact the college office for confirmation."
)
MATCH_MESSAGE = "Here is the information available in our college FAQ."

# Categories used (in this order) for the suggested-question buttons.
SUGGESTED_CATEGORIES = (
    "Admissions",
    "Courses",
    "Fees",
    "Scholarships",
    "Campus facilities",
    "Library",
)


def validate_unit_interval(name: str, value: object) -> float:
    """Return ``value`` as a float in [0, 1] or raise ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number between 0 and 1.")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number between 0 and 1.") from None
    if math.isnan(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a number between 0 and 1 (got {value!r}).")
    return number


def threshold_from_environment(default: float = DEFAULT_SIMILARITY_THRESHOLD) -> float:
    """Read the similarity threshold from the environment, else use ``default``."""
    raw = os.environ.get(THRESHOLD_ENV_VAR)
    if raw is None or not raw.strip():
        return validate_unit_interval("similarity threshold", default)
    try:
        return validate_unit_interval(THRESHOLD_ENV_VAR, raw.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid {THRESHOLD_ENV_VAR}: {exc}") from None
