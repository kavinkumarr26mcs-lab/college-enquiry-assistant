"""QueryProcessor: normalisation, retrieval, threshold, fallback, structure."""

import subprocess
import sys
from pathlib import Path

import pytest

from core.config import FALLBACK_MESSAGE
from core.faq_entry import FAQDataError, FAQEntry
from core.query_processor import QueryProcessor, normalize_text

RESULT_KEYS = {"matched", "faq_id", "question", "answer", "category", "similarity", "reason"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ normalisation
def test_normalize_lowercases_and_strips_punctuation_and_stop_words():
    assert normalize_text("What ARE the Library timings?!") == "library time"


def test_normalize_reduces_simple_plurals_and_uses_domain_lexicon():
    assert normalize_text("courses") == "course"
    assert normalize_text("Which programmes?") == "course"
    assert normalize_text("libraries") == "library"
    assert normalize_text("Wi-Fi") == "wifi"


@pytest.mark.parametrize("value", [None, 5, [], "", "   ", "?!", "the of and"])
def test_normalize_returns_empty_string_when_nothing_searchable(value):
    assert normalize_text(value) == ""


# ------------------------------------------------------------------ construction
def test_requires_at_least_one_entry():
    with pytest.raises(FAQDataError):
        QueryProcessor(entries=[])


def test_rejects_non_entry_objects():
    with pytest.raises(FAQDataError):
        QueryProcessor(entries=[{"faq_id": "FAQ001"}])


def test_missing_faq_file_raises_clear_error(tmp_path):
    with pytest.raises(FAQDataError, match="not found"):
        QueryProcessor(faq_path=tmp_path / "missing.json")


def test_question_with_no_searchable_words_is_rejected():
    entry = FAQEntry("FAQ001", "What is the?", "a", "Cat")
    with pytest.raises(FAQDataError, match="no searchable words"):
        QueryProcessor(entries=[entry])


@pytest.mark.parametrize("bad", [-0.1, 1.5, "abc", None, True, float("nan")])
def test_invalid_threshold_rejected(small_entries, bad):
    with pytest.raises(ValueError):
        QueryProcessor(entries=small_entries, threshold=bad)


def test_threshold_can_be_changed_and_is_validated(small_processor):
    small_processor.threshold = 0.5
    assert small_processor.threshold == 0.5
    with pytest.raises(ValueError):
        small_processor.threshold = 2


# ------------------------------------------------------------------ matching
def test_direct_question_matches_with_id_and_answer(small_processor):
    result = small_processor.process("What courses are available?")
    assert result["matched"] is True
    assert result["faq_id"] == "FAQ001"
    assert result["answer"] == "Synthetic courses answer."
    assert result["question"] == "What courses are available?"
    assert result["category"] == "Courses"
    assert result["reason"] == "reliable_match"
    assert result["similarity"] == pytest.approx(1.0)


def test_paraphrased_questions_match(small_processor):
    assert small_processor.process("Is there a hostel?")["faq_id"] == "FAQ002"
    assert small_processor.process("When does the library close?")["faq_id"] == "FAQ003"
    assert small_processor.process("How much are the tuition charges?")["faq_id"] == "FAQ004"


def test_matching_ignores_case_and_punctuation(small_processor):
    assert small_processor.process("WHAT COURSES ARE AVAILABLE???")["faq_id"] == "FAQ001"


@pytest.mark.parametrize(
    "question",
    ["What is the weather on Mars?", "Who won the cricket world cup?", "Tell me a joke."],
)
def test_out_of_scope_questions_trigger_fallback(small_processor, question):
    result = small_processor.process(question)
    assert result["matched"] is False
    assert result["faq_id"] is None
    assert result["answer"] == FALLBACK_MESSAGE


def test_one_shared_word_does_not_carry_an_unrelated_question(small_processor):
    result = small_processor.process("What is the price of gold in the market today?")
    assert result["matched"] is False
    assert result["reason"] == "below_threshold"


def test_raising_the_threshold_turns_a_match_into_a_fallback(small_entries):
    question = "Which degree programmes can I join?"
    lenient = QueryProcessor(entries=small_entries, threshold=0.2).process(question)
    strict = QueryProcessor(entries=small_entries, threshold=0.99).process(question)
    assert lenient["matched"] is True
    assert strict["matched"] is False
    assert strict["reason"] == "below_threshold"
    assert strict["similarity"] == lenient["similarity"] < 0.99


def test_near_tie_between_two_faqs_is_rejected_as_ambiguous():
    entries = [
        FAQEntry("FAQ001", "Library timings", "a", "Library"),
        FAQEntry("FAQ002", "Laboratory timings", "b", "Laboratories"),
    ]
    result = QueryProcessor(entries=entries, threshold=0.1).process("timings")
    assert result["matched"] is False
    assert result["reason"] == "ambiguous_match"


# ------------------------------------------------------------------ bad input
@pytest.mark.parametrize(
    "value, reason",
    [
        ("", "empty_input"),
        ("    ", "empty_input"),
        ("\n\t", "empty_input"),
        (None, "invalid_input"),
        (42, "invalid_input"),
        (["courses"], "invalid_input"),
        ({"q": "courses"}, "invalid_input"),
        ("x" * 501, "input_too_long"),
        ("?!?!", "no_known_terms"),
    ],
)
def test_empty_or_invalid_input_returns_fallback_without_raising(small_processor, value, reason):
    result = small_processor.process(value)
    assert result["matched"] is False
    assert result["reason"] == reason
    assert result["answer"] == FALLBACK_MESSAGE
    assert set(result) == RESULT_KEYS


def test_input_of_exactly_the_maximum_length_is_accepted(small_processor):
    assert small_processor.process("a" * 500)["reason"] != "input_too_long"


# ------------------------------------------------------------------ structure and stability
def test_result_structure_is_identical_for_match_and_fallback(small_processor):
    match = small_processor.process("What courses are available?")
    fallback = small_processor.process("What is the weather on Mars?")
    assert set(match) == set(fallback) == RESULT_KEYS
    assert isinstance(match["similarity"], float) and isinstance(fallback["similarity"], float)
    assert 0.0 <= match["similarity"] <= 1.0


def test_results_are_deterministic(small_processor):
    first = small_processor.process("Can I stay in a hostel?")
    assert all(small_processor.process("Can I stay in a hostel?") == first for _ in range(5))


def test_top_matches_orders_by_score(small_processor):
    top = small_processor.top_matches("library timings", k=2)
    assert len(top) == 2
    assert top[0]["faq_id"] == "FAQ003"
    assert top[0]["similarity"] >= top[1]["similarity"]
    assert small_processor.top_matches("", k=3) == []
    assert small_processor.top_matches("zzzz qqqq") == []


# ------------------------------------------------------------------ shipped dataset
def test_every_canonical_question_retrieves_its_own_faq(real_processor):
    wrong = []
    for entry in real_processor.entries:
        result = real_processor.process(entry.question)
        if result["faq_id"] != entry.faq_id:
            wrong.append((entry.faq_id, result["faq_id"], result["reason"]))
    assert wrong == []


def test_shipped_dataset_direct_and_out_of_scope_examples(real_processor):
    assert real_processor.process("What courses are available?")["faq_id"] == "FAQ007"
    assert real_processor.process("Is there a hostel?")["faq_id"] == "FAQ039"
    assert real_processor.process("What is the weather on Mars?")["matched"] is False


# ------------------------------------------------------------------ independence from Flask
def test_core_package_does_not_import_flask():
    code = "import sys, core; sys.exit(1 if 'flask' in sys.modules else 0)"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
