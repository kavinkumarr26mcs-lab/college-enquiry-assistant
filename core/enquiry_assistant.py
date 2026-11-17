"""EnquiryAssistant: the application-level facade over QueryProcessor.

It adds user-facing messages and error handling, and it never imports Flask.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    FALLBACK_MESSAGE,
    MATCH_MESSAGE,
    MAX_QUESTION_LENGTH,
    SUGGESTED_CATEGORIES,
)
from .query_processor import QueryProcessor

logger = logging.getLogger(__name__)

INPUT_ERROR_MESSAGES = {
    "empty_input": "Please enter a question.",
    "invalid_input": "The question must be text.",
    "input_too_long": (
        "Please keep your question under {max_length} characters."
    ),
}
INTERNAL_ERROR_MESSAGE = (
    "Something went wrong while processing your question. Please try again."
)


class EnquiryAssistant:
    """Receives questions, delegates retrieval and shapes the final response."""

    def __init__(
        self,
        faq_path: Union[str, Path, None] = None,
        threshold: Optional[float] = None,
        processor: Optional[QueryProcessor] = None,
    ) -> None:
        if processor is not None:
            self._processor = processor
        else:
            self._processor = QueryProcessor(
                faq_path=faq_path,
                threshold=DEFAULT_SIMILARITY_THRESHOLD if threshold is None else threshold,
            )

    @property
    def processor(self) -> QueryProcessor:
        return self._processor

    @property
    def threshold(self) -> float:
        return self._processor.threshold

    @property
    def faq_count(self) -> int:
        return len(self._processor.entries)

    # ------------------------------------------------------------------ responses
    @staticmethod
    def _response(
        *,
        success: bool,
        matched: bool,
        reason: str,
        message: str,
        faq_id: Optional[str] = None,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        category: Optional[str] = None,
        similarity: float = 0.0,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Every response has exactly these keys, whatever the outcome."""
        return {
            "success": success,
            "matched": matched,
            "faq_id": faq_id,
            "question": question,
            "answer": answer,
            "category": category,
            "similarity": similarity,
            "reason": reason,
            "message": message,
            "error": error,
        }

    def _failure(self, reason: str, text: str) -> Dict[str, Any]:
        return self._response(
            success=False, matched=False, reason=reason, message=text, error=text
        )

    def ask(self, question: Any) -> Dict[str, Any]:
        """Answer a question. Unexpected errors become a clean failure response."""
        try:
            result = self._processor.process(question)
        except Exception:  # noqa: BLE001 - the UI must always get a response
            logger.exception("Unexpected error while processing an enquiry")
            return self._failure("internal_error", INTERNAL_ERROR_MESSAGE)

        reason = result.get("reason", "")
        if reason in INPUT_ERROR_MESSAGES:
            text = INPUT_ERROR_MESSAGES[reason].format(max_length=MAX_QUESTION_LENGTH)
            return self._failure(reason, text)

        if result.get("matched"):
            return self._response(
                success=True,
                matched=True,
                reason=reason,
                message=MATCH_MESSAGE,
                faq_id=result["faq_id"],
                question=result["question"],
                answer=result["answer"],
                category=result.get("category"),
                similarity=result["similarity"],
            )
        return self._response(
            success=True,
            matched=False,
            reason=reason,
            message=FALLBACK_MESSAGE,
            answer=FALLBACK_MESSAGE,
            similarity=result.get("similarity", 0.0),
        )

    # ------------------------------------------------------------------ dataset views
    def list_faqs(self) -> List[Dict[str, Any]]:
        """All FAQs, from the same dataset the retrieval uses."""
        return [entry.to_dict() for entry in self._processor.entries]

    def suggested_questions(self, limit: int = 6) -> List[str]:
        """One canonical question per preferred category, topped up if needed."""
        entries = self._processor.entries
        first_by_category: Dict[str, str] = {}
        for entry in entries:
            first_by_category.setdefault(entry.category, entry.question)
        chosen: List[str] = [
            first_by_category[c] for c in SUGGESTED_CATEGORIES if c in first_by_category
        ]
        for category, question in first_by_category.items():
            if question not in chosen:
                chosen.append(question)
        return chosen[: max(int(limit), 0)]
