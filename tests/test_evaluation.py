"""Evaluation dataset validation and metric calculation."""

import json
from collections import Counter

import pytest

from core.faq_entry import load_faq_entries
from core.query_processor import normalize_text
from evaluation import evaluate as ev


def case(outcome, test_type="direct"):
    return {"outcome": outcome, "test_type": test_type}


# ------------------------------------------------------------------ shipped dataset
@pytest.fixture(scope="module")
def records(real_faq_path, real_evaluation_path):
    ids = [e.faq_id for e in load_faq_entries(real_faq_path)]
    return ev.load_evaluation_records(real_evaluation_path, ids)


def test_dataset_has_at_least_80_unique_questions(records):
    assert len(records) >= 80
    assert len({r["evaluation_id"] for r in records}) == len(records)


def test_dataset_distribution_meets_the_brief(records):
    counts = Counter(r["test_type"] for r in records)
    assert counts["direct"] >= 30
    assert counts["paraphrase"] >= 25
    assert counts["ambiguous"] >= 10
    assert counts["out_of_scope"] >= 15


def test_expected_outcomes_are_consistent(records):
    for r in records:
        if r["expected_outcome"] == "match":
            assert r["expected_faq_id"], r["evaluation_id"]
        else:
            assert r["expected_faq_id"] is None, r["evaluation_id"]
        if r["test_type"] in ("ambiguous", "out_of_scope"):
            assert r["expected_outcome"] == "fallback"


def test_every_faq_is_covered_by_at_least_one_question(records, real_faq_path):
    covered = {r["expected_faq_id"] for r in records if r["expected_faq_id"]}
    assert covered == {e.faq_id for e in load_faq_entries(real_faq_path)}


def test_questions_are_independent_of_the_faq_wording(records, real_faq_path):
    """No evaluation question may be a copy of an FAQ question or paraphrase."""
    known = set()
    for entry in load_faq_entries(real_faq_path):
        known.update(normalize_text(t) for t in entry.texts())
        known.update(t.casefold().strip(" ?.!") for t in entry.texts())
    for r in records:
        assert normalize_text(r["question"]) not in known, r["evaluation_id"]
        assert r["question"].casefold().strip(" ?.!") not in known, r["evaluation_id"]


# ------------------------------------------------------------------ validation rules
def valid(**overrides):
    record = {
        "evaluation_id": "EVAL001",
        "question": "Q one?",
        "expected_faq_id": "FAQ001",
        "expected_outcome": "match",
        "test_type": "direct",
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    "bad, message",
    [
        (valid(expected_outcome="maybe"), "expected_outcome"),
        (valid(test_type="weird"), "test_type"),
        (valid(expected_faq_id=None), "expected_faq_id"),
        (valid(expected_faq_id="FAQ999"), "unknown expected_faq_id"),
        (valid(expected_outcome="fallback"), "must have expected_faq_id null"),
        (valid(question="   "), "question"),
        ({"evaluation_id": "EVAL001"}, "missing"),
        ("not a dict", "JSON object"),
    ],
)
def test_validation_rejects_bad_records(bad, message):
    with pytest.raises(ev.EvaluationDataError, match=message):
        ev.validate_evaluation_records([bad], valid_faq_ids=["FAQ001"])


def test_validation_rejects_duplicate_ids_and_questions():
    with pytest.raises(ev.EvaluationDataError, match="duplicate evaluation_id"):
        ev.validate_evaluation_records([valid(), valid(question="Other?")])
    with pytest.raises(ev.EvaluationDataError, match="repeats the question"):
        ev.validate_evaluation_records([valid(), valid(evaluation_id="EVAL002")])


@pytest.mark.parametrize("payload", [[], {}, "x", None])
def test_validation_rejects_empty_or_wrong_shape(payload):
    with pytest.raises(ev.EvaluationDataError):
        ev.validate_evaluation_records(payload)


def test_loading_reports_missing_and_malformed_files(tmp_path, write_json):
    with pytest.raises(ev.EvaluationDataError, match="not found"):
        ev.load_evaluation_records(tmp_path / "none.json")
    with pytest.raises(ev.EvaluationDataError, match="not valid JSON"):
        ev.load_evaluation_records(write_json("bad.json", "[{", raw=True))


# ------------------------------------------------------------------ metric arithmetic
def test_metrics_match_hand_calculation():
    cases = (
        [case("correct_match")] * 4
        + [case("wrong_match")]
        + [case("missed_match")] * 2
        + [case("correct_fallback")] * 2
        + [case("false_match")]
    )
    m = ev.compute_metrics(cases)
    assert m["total"] == 10
    assert m["expected_matches"] == 7 and m["expected_fallbacks"] == 3
    assert m["correct_retrievals"] == 4
    assert m["incorrect_retrievals"] == 2  # wrong FAQ + FAQ given when fallback expected
    assert m["correct_fallbacks"] == 2 and m["incorrect_fallbacks"] == 2
    assert m["unmatched"] == 4
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["precision"] == pytest.approx(4 / 6)
    assert m["recall"] == pytest.approx(4 / 7)
    assert m["f1"] == pytest.approx(2 * (4 / 6) * (4 / 7) / ((4 / 6) + (4 / 7)))
    assert m["rejection_precision"] == pytest.approx(2 / 4)
    assert m["rejection_recall"] == pytest.approx(2 / 3)
    assert m["confusion"] == {
        "expected_match": {"predicted_correct_faq": 4, "predicted_wrong_faq": 1, "predicted_fallback": 2},
        "expected_fallback": {"predicted_faq": 1, "predicted_fallback": 2},
    }


def test_undefined_metrics_are_none_not_zero():
    only_fallbacks = ev.compute_metrics([case("correct_fallback")] * 3)
    assert only_fallbacks["precision"] is None
    assert only_fallbacks["recall"] is None
    assert only_fallbacks["f1"] is None
    assert only_fallbacks["accuracy"] == 1.0


def test_fallback_only_subset_reports_match_metrics_as_undefined():
    m = ev.compute_metrics([case("false_match"), case("correct_fallback")])
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None
    assert m["accuracy"] == 0.5
    assert m["rejection_recall"] == 0.5


def test_f1_is_zero_when_nothing_is_correct_but_defined():
    m = ev.compute_metrics([case("wrong_match"), case("missed_match")])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_empty_case_list_does_not_divide_by_zero():
    m = ev.compute_metrics([])
    assert m["total"] == 0 and m["accuracy"] is None


# ------------------------------------------------------------------ end-to-end runs
def test_run_is_repeatable_and_counts_add_up(real_processor, records):
    first = ev.run_evaluation(real_processor, records)
    second = ev.run_evaluation(real_processor, records)
    assert first == second
    m = first["metrics"]
    assert sum(m["counts"].values()) == m["total"] == len(records)
    assert sum(r["metrics"]["total"] for r in first["by_test_type"]) == m["total"]
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= m[key] <= 1.0


def test_classification_of_individual_results():
    match = {"expected_outcome": "match", "expected_faq_id": "FAQ001"}
    fallback = {"expected_outcome": "fallback", "expected_faq_id": None}
    assert ev.classify_case(match, {"matched": True, "faq_id": "FAQ001"}) == "correct_match"
    assert ev.classify_case(match, {"matched": True, "faq_id": "FAQ002"}) == "wrong_match"
    assert ev.classify_case(match, {"matched": False, "faq_id": None}) == "missed_match"
    assert ev.classify_case(fallback, {"matched": False, "faq_id": None}) == "correct_fallback"
    assert ev.classify_case(fallback, {"matched": True, "faq_id": "FAQ001"}) == "false_match"


def test_report_text_and_saved_json(real_processor, records, tmp_path):
    report = ev.run_evaluation(real_processor, records)
    text = ev.format_report(report, verbose=True)
    for label in ("Total questions", "Accuracy", "Precision", "Recall", "F1-score", "by test type"):
        assert label in text
    path = ev.save_report(report, tmp_path)
    assert path.parent == tmp_path
    assert json.loads(path.read_text(encoding="utf-8"))["metrics"]["total"] == len(records)


def test_command_line_runs_and_reports_errors(capsys, tmp_path):
    assert ev.main([]) == 0
    assert "Accuracy" in capsys.readouterr().out
    assert ev.main(["--questions", str(tmp_path / "missing.json")]) == 2
    assert "not found" in capsys.readouterr().err
    assert ev.main(["--threshold", "7"]) == 2
