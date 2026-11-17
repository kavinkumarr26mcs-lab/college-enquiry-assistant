"""Shared fixtures. Nothing here needs internet access."""

import json

import pytest

from core.config import DEFAULT_EVALUATION_PATH, DEFAULT_FAQ_PATH
from core.faq_entry import FAQEntry
from core.query_processor import QueryProcessor


@pytest.fixture
def write_json(tmp_path):
    """Write ``data`` (or raw text when ``raw=True``) to a temp file; return its path."""

    def _write(name, data, raw=False):
        path = tmp_path / name
        path.write_text(data if raw else json.dumps(data), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def small_records():
    return [
        {
            "faq_id": "FAQ001",
            "question": "What courses are available?",
            "answer": "Synthetic courses answer.",
            "category": "Courses",
            "paraphrases": ["Which programmes does the college offer?"],
        },
        {
            "faq_id": "FAQ002",
            "question": "Is hostel accommodation available?",
            "answer": "Synthetic hostel answer.",
            "category": "Campus facilities",
            "paraphrases": ["Can I stay on campus?"],
        },
        {
            "faq_id": "FAQ003",
            "question": "What are the library timings?",
            "answer": "Synthetic library answer.",
            "category": "Library",
            "paraphrases": ["When does the library open and close?"],
        },
        {
            "faq_id": "FAQ004",
            "question": "What is the fee structure?",
            "answer": "Synthetic fee answer.",
            "category": "Fees",
            "paraphrases": ["How much are the college fees?"],
        },
    ]


@pytest.fixture
def small_entries(small_records):
    return [FAQEntry.from_dict(record) for record in small_records]


@pytest.fixture
def small_processor(small_entries):
    return QueryProcessor(entries=small_entries, threshold=0.35)


@pytest.fixture(scope="session")
def real_processor():
    return QueryProcessor(faq_path=DEFAULT_FAQ_PATH)


@pytest.fixture(scope="session")
def real_faq_path():
    return DEFAULT_FAQ_PATH


@pytest.fixture(scope="session")
def real_evaluation_path():
    return DEFAULT_EVALUATION_PATH
