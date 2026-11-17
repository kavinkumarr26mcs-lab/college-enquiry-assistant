"""EnquiryAssistant: response shaping, fallback and error handling."""

import pytest

from core.config import FALLBACK_MESSAGE, MATCH_MESSAGE
from core.enquiry_assistant import INTERNAL_ERROR_MESSAGE, EnquiryAssistant
from core.faq_entry import FAQDataError

RESPONSE_KEYS = {
    "success", "matched", "faq_id", "question", "answer", "category",
    "similarity", "reason", "message", "error",
}


@pytest.fixture
def assistant(small_processor):
    return EnquiryAssistant(processor=small_processor)


def test_reliable_match_returns_id_and_answer_together(assistant):
    response = assistant.ask("What courses are available?")
    assert response["success"] is True
    assert response["matched"] is True
    assert response["faq_id"] == "FAQ001"
    assert response["answer"] == "Synthetic courses answer."
    assert response["message"] == MATCH_MESSAGE
    assert response["error"] is None


def test_unreliable_question_gets_the_fallback_not_an_invented_answer(assistant):
    response = assistant.ask("What is the weather on Mars?")
    assert response["success"] is True
    assert response["matched"] is False
    assert response["faq_id"] is None
    assert response["answer"] == FALLBACK_MESSAGE
    assert response["message"] == FALLBACK_MESSAGE


def test_fallback_text_matches_the_specification():
    assert FALLBACK_MESSAGE == (
        "I couldn't find a sufficiently reliable answer in the available FAQs. "
        "Please contact the college office for confirmation."
    )


@pytest.mark.parametrize(
    "value, reason",
    [("", "empty_input"), ("   ", "empty_input"), (None, "invalid_input"),
     (12, "invalid_input"), ("x" * 501, "input_too_long")],
)
def test_bad_input_is_a_failure_with_a_friendly_message(assistant, value, reason):
    response = assistant.ask(value)
    assert response["success"] is False
    assert response["matched"] is False
    assert response["reason"] == reason
    assert response["answer"] is None
    assert response["error"] and response["error"] == response["message"]


def test_response_keys_are_identical_in_every_case(assistant):
    cases = ["What courses are available?", "What is the weather on Mars?", "", None]
    assert all(set(assistant.ask(c)) == RESPONSE_KEYS for c in cases)


def test_unexpected_processor_errors_become_a_clean_failure(small_processor, monkeypatch):
    def boom(_question):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(small_processor, "process", boom)
    response = EnquiryAssistant(processor=small_processor).ask("anything")
    assert response["success"] is False
    assert response["reason"] == "internal_error"
    assert response["error"] == INTERNAL_ERROR_MESSAGE
    assert "secret" not in str(response)


def test_assistant_loads_from_a_file_path(write_json, small_records):
    path = write_json("faqs.json", small_records)
    loaded = EnquiryAssistant(faq_path=path, threshold=0.4)
    assert loaded.faq_count == 4
    assert loaded.threshold == 0.4
    assert loaded.ask("Is there a hostel?")["faq_id"] == "FAQ002"


def test_assistant_surfaces_missing_file_as_faq_data_error(tmp_path):
    with pytest.raises(FAQDataError):
        EnquiryAssistant(faq_path=tmp_path / "nope.json")


def test_list_faqs_comes_from_the_same_data(assistant, small_records):
    assert assistant.list_faqs() == [
        {**record, "paraphrases": list(record["paraphrases"])} for record in small_records
    ]


def test_suggested_questions_are_real_faq_questions(small_processor):
    assistant = EnquiryAssistant(processor=small_processor)
    questions = {e.question for e in small_processor.entries}
    suggestions = assistant.suggested_questions(3)
    assert len(suggestions) == 3
    assert set(suggestions) <= questions
    assert assistant.suggested_questions(0) == []


def test_every_suggested_question_gets_a_reliable_answer_on_real_data(real_faq_path):
    assistant = EnquiryAssistant(faq_path=real_faq_path)
    for question in assistant.suggested_questions(6):
        assert assistant.ask(question)["matched"] is True, question
