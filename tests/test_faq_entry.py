"""FAQEntry validation and FAQ file loading."""

import pytest

from core.faq_entry import FAQDataError, FAQEntry, load_faq_entries, parse_faq_records

REQUIRED_CATEGORIES = {
    "Admissions", "Courses", "Eligibility", "Academic procedures",
    "Examination procedures", "Attendance", "Library", "Laboratories",
    "Campus facilities", "Scholarships", "Fees", "Student support",
    "General college enquiries",
}


def make(**overrides):
    data = {
        "faq_id": "FAQ001",
        "question": "What courses are available?",
        "answer": "Synthetic answer.",
        "category": "Courses",
        "paraphrases": ["Which programmes are offered?"],
    }
    data.update(overrides)
    return data


# ------------------------------------------------------------------ the entry itself
def test_valid_entry_stores_all_fields():
    entry = FAQEntry.from_dict(make())
    assert entry.faq_id == "FAQ001"
    assert entry.question == "What courses are available?"
    assert entry.answer == "Synthetic answer."
    assert entry.category == "Courses"
    assert entry.paraphrases == ["Which programmes are offered?"]
    assert entry.texts() == [entry.question, "Which programmes are offered?"]


def test_paraphrases_are_optional():
    data = make()
    del data["paraphrases"]
    assert FAQEntry.from_dict(data).paraphrases == []


def test_text_is_trimmed_and_paraphrases_deduplicated():
    entry = FAQEntry.from_dict(
        make(
            question="  Q?  ",
            paraphrases=[" A ", "a", "Q?", "B"],
        )
    )
    assert entry.question == "Q?"
    assert entry.paraphrases == ["A", "B"]  # duplicate of A and of the question removed


@pytest.mark.parametrize("field", FAQEntry.REQUIRED_FIELDS)
def test_missing_required_field_is_reported(field):
    data = make()
    del data[field]
    with pytest.raises(FAQDataError, match=field):
        FAQEntry.from_dict(data)


@pytest.mark.parametrize("field", FAQEntry.REQUIRED_FIELDS)
@pytest.mark.parametrize("bad_value", ["", "   ", None, 5, ["x"]])
def test_blank_or_non_text_required_field_is_rejected(field, bad_value):
    with pytest.raises(FAQDataError):
        FAQEntry.from_dict(make(**{field: bad_value}))


@pytest.mark.parametrize("bad_id", ["faq001", "FAQ1", "001", "FAQ-001", "FAQABC"])
def test_faq_id_format_is_enforced(bad_id):
    with pytest.raises(FAQDataError, match="faq_id"):
        FAQEntry.from_dict(make(faq_id=bad_id))


@pytest.mark.parametrize("bad", ["a single string", 5, {"a": 1}, [1], [""]])
def test_invalid_paraphrases_rejected(bad):
    with pytest.raises(FAQDataError):
        FAQEntry.from_dict(make(paraphrases=bad))


def test_record_must_be_an_object():
    with pytest.raises(FAQDataError, match="JSON object"):
        FAQEntry.from_dict(["not", "a", "dict"])


def test_to_dict_round_trip():
    entry = FAQEntry.from_dict(make())
    assert FAQEntry.from_dict(entry.to_dict()) == entry


# ------------------------------------------------------------------ loading files
def test_valid_file_loads(write_json, small_records):
    entries = load_faq_entries(write_json("faqs.json", small_records))
    assert [e.faq_id for e in entries] == ["FAQ001", "FAQ002", "FAQ003", "FAQ004"]


def test_error_names_the_bad_record(write_json, small_records):
    del small_records[2]["answer"]
    with pytest.raises(FAQDataError, match=r"#3.*FAQ003.*answer|#3.*answer"):
        load_faq_entries(write_json("faqs.json", small_records))


def test_duplicate_faq_ids_rejected(write_json, small_records):
    small_records[1]["faq_id"] = "FAQ001"
    with pytest.raises(FAQDataError, match="duplicate FAQ ID.*FAQ001"):
        load_faq_entries(write_json("faqs.json", small_records))


def test_duplicate_questions_rejected(write_json, small_records):
    small_records[1]["question"] = small_records[0]["question"].upper()
    with pytest.raises(FAQDataError, match="repeats the question"):
        load_faq_entries(write_json("faqs.json", small_records))


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FAQDataError, match="not found"):
        load_faq_entries(tmp_path / "does_not_exist.json")


def test_directory_instead_of_file_gives_clear_error(tmp_path):
    with pytest.raises(FAQDataError):
        load_faq_entries(tmp_path)


def test_malformed_json_gives_clear_error(write_json):
    path = write_json("broken.json", '[{"faq_id": "FAQ001",', raw=True)
    with pytest.raises(FAQDataError, match="not valid JSON"):
        load_faq_entries(path)


def test_invalid_utf8_gives_clear_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(FAQDataError):
        load_faq_entries(path)


@pytest.mark.parametrize("payload", [{}, "text", 5, None])
def test_top_level_must_be_a_list(payload):
    with pytest.raises(FAQDataError, match="JSON list"):
        parse_faq_records(payload)


def test_empty_list_rejected():
    with pytest.raises(FAQDataError, match="empty"):
        parse_faq_records([])


# ------------------------------------------------------------------ the shipped dataset
def test_shipped_dataset_loads_with_at_least_60_unique_entries(real_faq_path):
    entries = load_faq_entries(real_faq_path)
    assert len(entries) >= 60
    assert len({e.faq_id for e in entries}) == len(entries)
    assert len({e.question.casefold() for e in entries}) == len(entries)


def test_shipped_dataset_covers_every_required_category(real_faq_path):
    categories = {e.category for e in load_faq_entries(real_faq_path)}
    assert REQUIRED_CATEGORIES <= categories


def test_shipped_dataset_ids_are_sequential(real_faq_path):
    ids = [e.faq_id for e in load_faq_entries(real_faq_path)]
    assert ids == [f"FAQ{i:03d}" for i in range(1, len(ids) + 1)]


def test_no_shipped_answer_is_a_placeholder(real_faq_path):
    for entry in load_faq_entries(real_faq_path):
        assert "placeholder" not in entry.answer.lower(), entry.faq_id
        assert "synthetic" not in entry.answer.lower(), entry.faq_id


def test_every_shipped_answer_is_sourced_or_defers_to_the_office(real_faq_path):
    for entry in load_faq_entries(real_faq_path):
        assert "Handbook 2024-25" in entry.answer or "info@vetias.ac.in" in entry.answer, entry.faq_id


def test_every_shipped_entry_has_paraphrases(real_faq_path):
    assert all(e.paraphrases for e in load_faq_entries(real_faq_path))
