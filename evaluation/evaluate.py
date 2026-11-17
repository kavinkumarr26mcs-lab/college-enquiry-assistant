"""Repeatable evaluation of the core retrieval system (no Flask required).

Run from the project root:

    python -m evaluation.evaluate
    python -m evaluation.evaluate --threshold 0.4 --verbose --save

Outcome of every evaluation question
------------------------------------
correct_match     match expected, system returned the expected FAQ
wrong_match       match expected, system returned a different FAQ
missed_match      match expected, system returned the fallback
correct_fallback  fallback expected, system returned the fallback
false_match       fallback expected, system returned an FAQ answer

Metric definitions (all computed from these counts)
---------------------------------------------------
accuracy   (correct_match + correct_fallback) / total
precision  correct_match / (correct_match + wrong_match + false_match)
           Of the FAQ answers the system returned, how many were right.
           Undefined (None) for a subset that has no expected FAQ answers.
recall     correct_match / (correct_match + wrong_match + missed_match)
           Of the questions that have an FAQ answer, how many got the right one.
f1         2 * precision * recall / (precision + recall)
rejection_precision  correct_fallback / (correct_fallback + missed_match)
rejection_recall     correct_fallback / (correct_fallback + false_match)

A metric whose denominator is zero is reported as ``None`` ("n/a"), never as 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

if __package__ in (None, ""):  # allow `python evaluation/evaluate.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import (  # noqa: E402
    DEFAULT_EVALUATION_PATH,
    DEFAULT_FAQ_PATH,
    RESULTS_DIR,
    threshold_from_environment,
)
from core.faq_entry import FAQDataError  # noqa: E402
from core.query_processor import QueryProcessor, normalize_text  # noqa: E402

ALLOWED_TEST_TYPES = ("direct", "paraphrase", "conversational", "ambiguous", "out_of_scope")
ALLOWED_OUTCOMES = ("match", "fallback")
REQUIRED_FIELDS = (
    "evaluation_id",
    "question",
    "expected_faq_id",
    "expected_outcome",
    "test_type",
)
OUTCOMES = ("correct_match", "wrong_match", "missed_match", "correct_fallback", "false_match")


class EvaluationDataError(ValueError):
    """Raised when the evaluation dataset is missing, malformed or inconsistent."""


# ---------------------------------------------------------------------- loading
def validate_evaluation_records(
    records: Any, valid_faq_ids: Optional[Iterable[str]] = None
) -> List[Dict[str, Any]]:
    """Check the dataset and return it; raise ``EvaluationDataError`` listing problems."""
    if not isinstance(records, list) or not records:
        raise EvaluationDataError("Evaluation data must be a non-empty JSON list.")
    known_ids = set(valid_faq_ids) if valid_faq_ids is not None else None
    problems: List[str] = []
    seen_ids: Dict[str, int] = {}
    seen_questions: Dict[str, str] = {}

    for position, record in enumerate(records, 1):
        label = f"record #{position}"
        if not isinstance(record, dict):
            problems.append(f"{label}: must be a JSON object")
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in record]
        if missing:
            problems.append(f"{label}: missing {', '.join(missing)}")
            continue

        eval_id, question = record["evaluation_id"], record["question"]
        outcome, faq_id = record["expected_outcome"], record["expected_faq_id"]
        if not isinstance(eval_id, str) or not eval_id.strip():
            problems.append(f"{label}: evaluation_id must be non-empty text")
        else:
            label = eval_id
            if eval_id in seen_ids:
                problems.append(f"{label}: duplicate evaluation_id")
            seen_ids[eval_id] = position
        if not isinstance(question, str) or not question.strip():
            problems.append(f"{label}: question must be non-empty text")
        else:
            key = question.strip().casefold()
            if key in seen_questions:
                problems.append(f"{label}: repeats the question of {seen_questions[key]}")
            seen_questions[key] = label
        if record["test_type"] not in ALLOWED_TEST_TYPES:
            problems.append(f"{label}: test_type must be one of {ALLOWED_TEST_TYPES}")
        if outcome not in ALLOWED_OUTCOMES:
            problems.append(f"{label}: expected_outcome must be 'match' or 'fallback'")
        elif outcome == "match":
            if not isinstance(faq_id, str) or not faq_id:
                problems.append(f"{label}: a 'match' needs an expected_faq_id")
            elif known_ids is not None and faq_id not in known_ids:
                problems.append(f"{label}: unknown expected_faq_id {faq_id}")
        elif faq_id is not None:
            problems.append(f"{label}: a 'fallback' must have expected_faq_id null")

    if problems:
        shown = "; ".join(problems[:10])
        extra = f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""
        raise EvaluationDataError(f"Invalid evaluation data: {shown}{extra}")
    return records


def load_evaluation_records(
    path: Any = None, valid_faq_ids: Optional[Iterable[str]] = None
) -> List[Dict[str, Any]]:
    file_path = Path(path or DEFAULT_EVALUATION_PATH)
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise EvaluationDataError(f"Evaluation file not found: {file_path}") from None
    except json.JSONDecodeError as exc:
        raise EvaluationDataError(
            f"Evaluation file is not valid JSON ({file_path.name}, line {exc.lineno}): {exc.msg}"
        ) from None
    except (UnicodeDecodeError, OSError) as exc:
        raise EvaluationDataError(f"Could not read evaluation file {file_path}: {exc}") from None
    return validate_evaluation_records(raw, valid_faq_ids)


# ---------------------------------------------------------------------- scoring
def classify_case(record: Dict[str, Any], result: Dict[str, Any]) -> str:
    predicted = bool(result.get("matched"))
    if record["expected_outcome"] == "match":
        if not predicted:
            return "missed_match"
        return "correct_match" if result.get("faq_id") == record["expected_faq_id"] else "wrong_match"
    return "false_match" if predicted else "correct_fallback"


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator == 0 else numerator / denominator


def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-question outcomes into the metrics defined in the module docs."""
    counts = {name: 0 for name in OUTCOMES}
    for case in cases:
        counts[case["outcome"]] += 1
    tp, wrong, missed = counts["correct_match"], counts["wrong_match"], counts["missed_match"]
    correct_fb, false_m = counts["correct_fallback"], counts["false_match"]
    total = len(cases)
    # With no expected FAQ answers (e.g. only out-of-scope questions) precision
    # would be a trivial 0, so match metrics are reported as undefined instead.
    has_expected_matches = (tp + wrong + missed) > 0
    precision = _ratio(tp, tp + wrong + false_m) if has_expected_matches else None
    recall = _ratio(tp, tp + wrong + missed)
    return {
        "total": total,
        "expected_matches": tp + wrong + missed,
        "expected_fallbacks": correct_fb + false_m,
        "counts": counts,
        "correct_retrievals": tp,
        "incorrect_retrievals": wrong + false_m,
        "correct_fallbacks": correct_fb,
        "incorrect_fallbacks": missed,
        "unmatched": missed + correct_fb,
        "accuracy": _ratio(tp + correct_fb, total),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "rejection_precision": _ratio(correct_fb, correct_fb + missed),
        "rejection_recall": _ratio(correct_fb, correct_fb + false_m),
        "confusion": {
            "expected_match": {
                "predicted_correct_faq": tp,
                "predicted_wrong_faq": wrong,
                "predicted_fallback": missed,
            },
            "expected_fallback": {
                "predicted_faq": false_m,
                "predicted_fallback": correct_fb,
            },
        },
    }


def run_evaluation(processor: Any, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Run every question through ``processor.process`` and build the report."""
    cases: List[Dict[str, Any]] = []
    for record in records:
        result = processor.process(record["question"])
        cases.append(
            {
                "evaluation_id": record["evaluation_id"],
                "question": record["question"],
                "test_type": record["test_type"],
                "expected_outcome": record["expected_outcome"],
                "expected_faq_id": record["expected_faq_id"],
                "predicted_matched": bool(result.get("matched")),
                "predicted_faq_id": result.get("faq_id"),
                "similarity": result.get("similarity"),
                "reason": result.get("reason"),
                "outcome": classify_case(record, result),
            }
        )
    by_type = []
    for test_type in ALLOWED_TEST_TYPES:
        subset = [c for c in cases if c["test_type"] == test_type]
        if subset:
            by_type.append({"test_type": test_type, "metrics": compute_metrics(subset)})
    return {
        "threshold": getattr(processor, "threshold", None),
        "faq_count": len(getattr(processor, "entries", [])),
        "metrics": compute_metrics(cases),
        "by_test_type": by_type,
        "cases": cases,
        "failures": [c for c in cases if c["outcome"] in ("wrong_match", "missed_match", "false_match")],
    }


def build_report(
    processor: Any = None,
    evaluation_path: Any = None,
    faq_path: Any = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Load data, run the evaluation and return the report dictionary."""
    if processor is None:
        processor = QueryProcessor(
            faq_path=faq_path or DEFAULT_FAQ_PATH,
            threshold=threshold_from_environment() if threshold is None else threshold,
        )
    records = load_evaluation_records(evaluation_path, processor.faq_ids)
    return run_evaluation(processor, records)


# ---------------------------------------------------------------------- output
def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_report(report: Dict[str, Any], verbose: bool = False) -> str:
    m = report["metrics"]
    lines = [
        "College Enquiry Assistant - evaluation report",
        "=" * 46,
        f"FAQs loaded            : {report['faq_count']}",
        f"Similarity threshold   : {report['threshold']}",
        f"Total questions        : {m['total']}",
        f"Expected FAQ matches   : {m['expected_matches']}",
        f"Expected fallbacks     : {m['expected_fallbacks']}",
        "",
        f"Correct FAQ retrievals   : {m['correct_retrievals']}",
        f"Incorrect FAQ retrievals : {m['incorrect_retrievals']}"
        f"  (wrong FAQ {m['counts']['wrong_match']}, FAQ returned when fallback expected {m['counts']['false_match']})",
        f"Correct fallbacks        : {m['correct_fallbacks']}",
        f"Incorrect fallbacks      : {m['incorrect_fallbacks']}  (fallback returned when an FAQ was expected)",
        "",
        f"Accuracy  : {_fmt(m['accuracy'])}",
        f"Precision : {_fmt(m['precision'])}",
        f"Recall    : {_fmt(m['recall'])}",
        f"F1-score  : {_fmt(m['f1'])}",
        f"Rejection precision / recall : {_fmt(m['rejection_precision'])} / {_fmt(m['rejection_recall'])}",
        "",
        "Results by test type",
        "-" * 46,
        f"{'type':<16}{'n':>4}{'correct':>9}{'accuracy':>10}{'precision':>11}{'recall':>8}",
    ]
    for row in report["by_test_type"]:
        t = row["metrics"]
        correct = t["correct_retrievals"] + t["correct_fallbacks"]
        lines.append(
            f"{row['test_type']:<16}{t['total']:>4}{correct:>9}{_fmt(t['accuracy']):>10}"
            f"{_fmt(t['precision']):>11}{_fmt(t['recall']):>8}"
        )
    if verbose and report["failures"]:
        lines += ["", "Incorrect cases", "-" * 46]
        for case in report["failures"]:
            lines.append(
                f"{case['evaluation_id']} [{case['test_type']}] {case['outcome']}: "
                f"{case['question']!r} expected={case['expected_faq_id']} "
                f"got={case['predicted_faq_id']} sim={case['similarity']} reason={case['reason']}"
            )
    lines += [
        "",
        "Definitions: precision = correct FAQ answers / FAQ answers returned;",
        "recall = correct FAQ answers / questions that have an FAQ answer;",
        "n/a = denominator is zero. See evaluation/evaluate.py for full definitions.",
    ]
    return "\n".join(lines)


def save_report(report: Dict[str, Any], directory: Any = None) -> Path:
    target = Path(directory or RESULTS_DIR)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the FAQ retrieval system.")
    parser.add_argument("--faqs", default=str(DEFAULT_FAQ_PATH), help="FAQ JSON file")
    parser.add_argument("--questions", default=str(DEFAULT_EVALUATION_PATH), help="evaluation JSON file")
    parser.add_argument("--threshold", type=float, default=None, help="similarity threshold (0-1)")
    parser.add_argument("--verbose", action="store_true", help="list every incorrect case")
    parser.add_argument("--save", action="store_true", help="write the report as JSON to evaluation/results/")
    args = parser.parse_args(argv)
    try:
        report = build_report(
            evaluation_path=args.questions, faq_path=args.faqs, threshold=args.threshold
        )
    except (FAQDataError, EvaluationDataError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(format_report(report, verbose=args.verbose))
    if args.save:
        print(f"\nSaved: {save_report(report)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
