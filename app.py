"""Flask web application for the College Enquiry Assistant.

All retrieval and fallback decisions are made by the Flask-free ``core``
package; this module only validates HTTP input and renders pages.

Run locally:  python app.py   then open http://127.0.0.1:5000
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, request

from core.config import (
    DEFAULT_EVALUATION_PATH,
    DEFAULT_FAQ_PATH,
    MAX_QUESTION_LENGTH,
    threshold_from_environment,
)
from core.enquiry_assistant import EnquiryAssistant
from core.faq_entry import FAQDataError
from evaluation.evaluate import EvaluationDataError, build_report

logger = logging.getLogger(__name__)

COLLEGE_NAME = "VET Institute of Arts and Science"
COLLEGE_DESCRIPTOR = "(Co-education) College, Thindal, Erode, Tamil Nadu"
PROJECT_NAME = "College Enquiry Assistant"
PROJECT_CODE = "G04"

CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' https://fonts.googleapis.com",
        "font-src https://fonts.gstatic.com",
        "img-src 'self' data:",
        "connect-src 'self'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ]
)

SERVICE_UNAVAILABLE_MESSAGE = (
    "The FAQ data could not be loaded, so enquiries are unavailable right now."
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Application factory (also used by the automated tests)."""
    app = Flask(__name__)
    app.json.sort_keys = False
    app.config.update(
        FAQ_PATH=str(DEFAULT_FAQ_PATH),
        EVALUATION_PATH=str(DEFAULT_EVALUATION_PATH),
        SIMILARITY_THRESHOLD=None,  # None -> environment variable / default
        ENABLE_EVALUATION=_env_flag("ENQUIRY_ENABLE_EVALUATION", True),
        MAX_CONTENT_LENGTH=16 * 1024,
    )
    if test_config:
        app.config.update(test_config)

    threshold = app.config["SIMILARITY_THRESHOLD"]
    if threshold is None:
        threshold = threshold_from_environment()  # raises ValueError if invalid

    app.config["STARTUP_ERROR"] = None
    assistant: Optional[EnquiryAssistant] = None
    try:
        assistant = EnquiryAssistant(faq_path=app.config["FAQ_PATH"], threshold=threshold)
    except FAQDataError as exc:
        app.config["STARTUP_ERROR"] = str(exc)
        logger.error("FAQ data could not be loaded: %s", exc)
    app.extensions["enquiry_assistant"] = assistant

    # ------------------------------------------------------------- template helpers
    @app.context_processor
    def inject_globals() -> Dict[str, Any]:
        return {
            "college_name": COLLEGE_NAME,
            "college_descriptor": COLLEGE_DESCRIPTOR,
            "project_name": PROJECT_NAME,
            "project_code": PROJECT_CODE,
            "evaluation_enabled": app.config["ENABLE_EVALUATION"],
            "current_year": datetime.now().year,
        }

    @app.template_filter("score")
    def score_filter(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def get_assistant() -> Optional[EnquiryAssistant]:
        return app.extensions["enquiry_assistant"]

    def unavailable_page():
        return (
            render_template(
                "error.html",
                active_page="",
                status=503,
                title="Service unavailable",
                message=SERVICE_UNAVAILABLE_MESSAGE,
                detail=app.config["STARTUP_ERROR"],
            ),
            503,
        )

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return render_template("index.html", active_page="home")

    @app.get("/enquiry")
    def enquiry():
        assistant = get_assistant()
        if assistant is None:
            return unavailable_page()
        return render_template(
            "enquiry.html",
            active_page="enquiry",
            suggestions=assistant.suggested_questions(6),
            max_length=MAX_QUESTION_LENGTH,
        )

    @app.get("/about")
    def about():
        assistant = get_assistant()
        return render_template(
            "about.html",
            active_page="about",
            threshold=assistant.threshold if assistant else None,
            faq_count=assistant.faq_count if assistant else None,
        )

    @app.get("/faq")
    def faq():
        assistant = get_assistant()
        if assistant is None:
            return unavailable_page()
        faqs = assistant.list_faqs()
        categories = sorted({item["category"] for item in faqs})
        return render_template(
            "faq.html", active_page="faq", faqs=faqs, categories=categories
        )

    @app.get("/evaluation")
    def evaluation():
        if not app.config["ENABLE_EVALUATION"]:
            return error_page(404)
        assistant = get_assistant()
        if assistant is None:
            return unavailable_page()
        report, error = None, None
        try:
            report = build_report(
                processor=assistant.processor,
                evaluation_path=app.config["EVALUATION_PATH"],
            )
        except EvaluationDataError as exc:
            error = str(exc)
        return render_template(
            "evaluation.html", active_page="evaluation", report=report, error=error
        )

    # ------------------------------------------------------------------ JSON API
    def api_error(status: int, reason: str, message: str):
        body = {
            "success": False,
            "matched": False,
            "faq_id": None,
            "question": None,
            "answer": None,
            "category": None,
            "similarity": 0.0,
            "reason": reason,
            "message": message,
            "error": message,
        }
        return jsonify(body), status

    @app.post("/api/enquire")
    def api_enquire():
        assistant = get_assistant()
        if assistant is None:
            return api_error(503, "service_unavailable", SERVICE_UNAVAILABLE_MESSAGE)
        if not request.is_json:
            return api_error(415, "unsupported_media_type", "Send the request as JSON.")
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return api_error(400, "invalid_json", "The request body must be a JSON object.")
        if "question" not in payload:
            return api_error(400, "missing_question", "The 'question' field is required.")

        result = assistant.ask(payload["question"])
        if result["success"]:
            status = 200
        elif result["reason"] == "internal_error":
            status = 500
        else:
            status = 400
        return jsonify(result), status

    @app.get("/api/evaluation")
    def api_evaluation():
        if not app.config["ENABLE_EVALUATION"]:
            return api_error(404, "not_found", "The evaluation interface is disabled.")
        assistant = get_assistant()
        if assistant is None:
            return api_error(503, "service_unavailable", SERVICE_UNAVAILABLE_MESSAGE)
        try:
            report = build_report(
                processor=assistant.processor,
                evaluation_path=app.config["EVALUATION_PATH"],
            )
        except EvaluationDataError as exc:
            return api_error(500, "evaluation_data_error", str(exc))
        return jsonify({"success": True, "report": report})

    # ------------------------------------------------------------------ errors
    def error_page(status: int):
        messages = {
            404: ("Page not found", "The page you are looking for does not exist."),
            405: ("Method not allowed", "That action is not supported for this page."),
            413: ("Request too large", "The request was larger than allowed."),
            500: ("Something went wrong", "An unexpected error occurred. Please try again."),
        }
        title, message = messages.get(status, ("Error", "Something went wrong."))
        return (
            render_template(
                "error.html",
                active_page="",
                status=status,
                title=title,
                message=message,
                detail=None,
            ),
            status,
        )

    def handle_http_error(status: int):
        if request.path.startswith("/api/"):
            reasons = {
                404: "not_found",
                405: "method_not_allowed",
                413: "request_too_large",
                500: "internal_error",
            }
            return api_error(status, reasons.get(status, "error"), f"Error {status}.")
        return error_page(status)

    for code in (404, 405, 413, 500):
        app.register_error_handler(code, lambda _exc, code=code: handle_http_error(code))

    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=_env_flag("FLASK_DEBUG", False),
    )
