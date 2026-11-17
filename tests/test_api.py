"""Flask routes and the JSON API."""

import json

import pytest

from app import create_app

PAGES = ["/", "/enquiry", "/about", "/faq", "/evaluation"]


@pytest.fixture
def app():
    return create_app({"TESTING": True})


@pytest.fixture
def client(app):
    return app.test_client()


def post(client, payload):
    return client.post("/api/enquire", json=payload)


@pytest.mark.parametrize("path", PAGES)
def test_every_page_renders(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.data


def test_home_page_has_required_content(client):
    html = client.get("/").get_data(as_text=True)
    assert "Your College, One Conversation Away." in html
    assert "Ask an Enquiry" in html
    assert "Instant Answers" in html and "FAQ References" in html and "Guided Enquiries" in html


def test_faq_page_lists_every_faq_from_the_same_dataset(client, app):
    html = client.get("/faq").get_data(as_text=True)
    for item in app.extensions["enquiry_assistant"].list_faqs():
        assert item["faq_id"] in html
    assert 'id="faq-empty"' in html and "No FAQs found" in html


def test_enquiry_page_has_chat_controls(client):
    html = client.get("/enquiry").get_data(as_text=True)
    for marker in ('id="chat-log"', 'id="question-input"', 'id="send-button"',
                   'id="reset-button"', 'class="chip"', "welcome-template"):
        assert marker in html


def test_evaluation_page_shows_metrics(client):
    html = client.get("/evaluation").get_data(as_text=True)
    for label in ("Total questions", "Precision", "Recall", "F1-score", "Confusion counts"):
        assert label in html


def test_evaluation_can_be_disabled():
    client = create_app({"TESTING": True, "ENABLE_EVALUATION": False}).test_client()
    assert client.get("/evaluation").status_code == 404
    assert client.get("/api/evaluation").status_code == 404
    assert b"Evaluation (development)" not in client.get("/").data


def test_evaluation_page_reports_bad_dataset_instead_of_crashing(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[{", encoding="utf-8")
    client = create_app({"TESTING": True, "EVALUATION_PATH": str(bad)}).test_client()
    response = client.get("/evaluation")
    assert response.status_code == 200
    assert b"Evaluation unavailable" in response.data


# ------------------------------------------------------------------ /api/enquire
def test_matched_enquiry_returns_id_and_answer(client):
    response = post(client, {"question": "What courses are available?"})
    body = response.get_json()
    assert response.status_code == 200
    assert body["success"] is True and body["matched"] is True
    assert body["faq_id"] == "FAQ007"
    assert "B.Sc. Computer Science" in body["answer"]
    assert 0 < body["similarity"] <= 1


def test_fallback_enquiry_is_a_successful_unmatched_response(client):
    response = post(client, {"question": "What is the weather on Mars?"})
    body = response.get_json()
    assert response.status_code == 200
    assert body["success"] is True and body["matched"] is False
    assert body["faq_id"] is None
    assert "couldn't find a sufficiently reliable answer" in body["answer"]


def test_response_structure_is_stable(client):
    keys = {
        "success", "matched", "faq_id", "question", "answer", "category",
        "similarity", "reason", "message", "error",
    }
    for payload in (
        {"question": "What courses are available?"},
        {"question": "What is the weather on Mars?"},
        {"question": ""},
        {},
    ):
        assert set(post(client, payload).get_json()) == keys


@pytest.mark.parametrize("question", ["", "   ", 5, None, ["a"], "x" * 501])
def test_invalid_questions_return_400(client, question):
    response = post(client, {"question": question})
    assert response.status_code == 400
    assert response.get_json()["success"] is False
    assert response.get_json()["error"]


def test_missing_question_field_returns_400(client):
    response = post(client, {"q": "hello"})
    assert response.status_code == 400
    assert response.get_json()["reason"] == "missing_question"


def test_non_object_json_returns_400(client):
    assert post(client, ["question"]).status_code == 400
    assert post(client, "question").status_code == 400


def test_malformed_json_returns_400(client):
    response = client.post("/api/enquire", data="{oops", content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["reason"] == "invalid_json"


def test_non_json_content_type_returns_415(client):
    response = client.post("/api/enquire", data="question=hi", content_type="text/plain")
    assert response.status_code == 415


def test_wrong_method_returns_json_405(client):
    response = client.get("/api/enquire")
    assert response.status_code == 405
    assert response.get_json()["success"] is False


def test_oversized_body_returns_413(client):
    response = client.post(
        "/api/enquire", data=json.dumps({"question": "a" * 20000}), content_type="application/json"
    )
    assert response.status_code == 413


def test_unknown_page_returns_404_html_and_unknown_api_returns_json(client):
    assert client.get("/missing").status_code == 404
    assert b"Page not found" in client.get("/missing").data
    response = client.get("/api/missing")
    assert response.status_code == 404 and response.get_json()["reason"] == "not_found"


def test_internal_errors_return_500_without_details(app, client, monkeypatch):
    def boom(_q):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(app.extensions["enquiry_assistant"].processor, "process", boom)
    response = post(client, {"question": "anything"})
    assert response.status_code == 500
    assert "secret" not in response.get_data(as_text=True)


# ------------------------------------------------------------------ safety and configuration
def test_submitted_text_is_never_echoed_back(client):
    payload = "<script>alert(1)</script> courses"
    response = post(client, {"question": payload})
    assert "<script>" not in response.get_data(as_text=True)


def test_faq_text_is_html_escaped_in_pages(tmp_path):
    path = tmp_path / "faqs.json"
    path.write_text(
        json.dumps([{
            "faq_id": "FAQ001", "question": "Is <b>this</b> safe?",
            "answer": "<script>alert('x')</script>", "category": "Courses",
        }]),
        encoding="utf-8",
    )
    client = create_app({"TESTING": True, "FAQ_PATH": str(path)}).test_client()
    html = client.get("/faq").get_data(as_text=True)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_security_headers_are_sent(client):
    headers = client.get("/").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert client.post("/api/enquire", json={"question": "x"}).headers["Cache-Control"] == "no-store"


def test_missing_faq_file_gives_503_not_a_crash(tmp_path):
    client = create_app({"TESTING": True, "FAQ_PATH": str(tmp_path / "gone.json")}).test_client()
    assert client.get("/").status_code == 200
    assert client.get("/enquiry").status_code == 503
    assert client.get("/faq").status_code == 503
    response = post(client, {"question": "courses"})
    assert response.status_code == 503 and response.get_json()["reason"] == "service_unavailable"


def test_malformed_faq_file_gives_503(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[{", encoding="utf-8")
    client = create_app({"TESTING": True, "FAQ_PATH": str(path)}).test_client()
    assert post(client, {"question": "courses"}).status_code == 503


def test_threshold_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ENQUIRY_SIMILARITY_THRESHOLD", "0.9")
    app = create_app({"TESTING": True})
    assert app.extensions["enquiry_assistant"].threshold == 0.9
    assert "0.90" in app.test_client().get("/about").get_data(as_text=True)


def test_invalid_threshold_in_the_environment_fails_loudly(monkeypatch):
    monkeypatch.setenv("ENQUIRY_SIMILARITY_THRESHOLD", "banana")
    with pytest.raises(ValueError, match="ENQUIRY_SIMILARITY_THRESHOLD"):
        create_app({"TESTING": True})


def test_explicit_threshold_config_overrides_the_default():
    app = create_app({"TESTING": True, "SIMILARITY_THRESHOLD": 0.6})
    assert app.extensions["enquiry_assistant"].threshold == 0.6
