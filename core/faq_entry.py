"""FAQEntry: one validated FAQ record, plus safe loading of the FAQ file."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


class FAQDataError(ValueError):
    """Raised when FAQ data is missing, malformed or invalid."""


FAQ_ID_PATTERN = re.compile(r"^FAQ\d{3,}$")


class FAQEntry:
    """A single FAQ: identifier, canonical question, answer, category, paraphrases."""

    REQUIRED_FIELDS = ("faq_id", "question", "answer", "category")

    def __init__(
        self,
        faq_id: str,
        question: str,
        answer: str,
        category: str,
        paraphrases: Optional[Sequence[str]] = None,
    ) -> None:
        self.faq_id = self._require_text("faq_id", faq_id)
        self.question = self._require_text("question", question)
        self.answer = self._require_text("answer", answer)
        self.category = self._require_text("category", category)
        if not FAQ_ID_PATTERN.match(self.faq_id):
            raise FAQDataError(
                f"'faq_id' must look like FAQ001 (got {self.faq_id!r})."
            )
        self.paraphrases: List[str] = self._clean_paraphrases(paraphrases)

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _require_text(name: str, value: Any) -> str:
        if not isinstance(value, str):
            raise FAQDataError(f"'{name}' must be text.")
        cleaned = value.strip()
        if not cleaned:
            raise FAQDataError(f"'{name}' must not be empty.")
        return cleaned

    def _clean_paraphrases(self, paraphrases: Any) -> List[str]:
        if paraphrases is None:
            return []
        if isinstance(paraphrases, (str, bytes)) or not isinstance(
            paraphrases, (list, tuple)
        ):
            raise FAQDataError("'paraphrases' must be a list of text values.")
        seen = {self.question.casefold()}
        cleaned: List[str] = []
        for item in paraphrases:
            text = self._require_text("paraphrases item", item)
            key = text.casefold()
            if key not in seen:
                seen.add(key)
                cleaned.append(text)
        return cleaned

    # ------------------------------------------------------------------ conversion
    @classmethod
    def from_dict(cls, data: Any, position: Optional[int] = None) -> "FAQEntry":
        label = f"FAQ record #{position}" if position is not None else "FAQ record"
        if not isinstance(data, dict):
            raise FAQDataError(f"{label}: each record must be a JSON object.")
        missing = [name for name in cls.REQUIRED_FIELDS if name not in data]
        if missing:
            raise FAQDataError(
                f"{label}: missing required field(s): {', '.join(missing)}."
            )
        try:
            return cls(
                faq_id=data["faq_id"],
                question=data["question"],
                answer=data["answer"],
                category=data["category"],
                paraphrases=data.get("paraphrases"),
            )
        except FAQDataError as exc:
            ident = data.get("faq_id") if isinstance(data.get("faq_id"), str) else None
            prefix = f"{label} ({ident})" if ident else label
            raise FAQDataError(f"{prefix}: {exc}") from None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faq_id": self.faq_id,
            "question": self.question,
            "answer": self.answer,
            "category": self.category,
            "paraphrases": list(self.paraphrases),
        }

    def texts(self) -> List[str]:
        """The canonical question followed by its approved paraphrases."""
        return [self.question, *self.paraphrases]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FAQEntry) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(self.faq_id)

    def __repr__(self) -> str:
        return f"FAQEntry(faq_id={self.faq_id!r}, question={self.question!r})"


def parse_faq_records(raw: Any, source: str = "FAQ data") -> List[FAQEntry]:
    """Validate decoded JSON and return FAQ entries (unique IDs and questions)."""
    if not isinstance(raw, list):
        raise FAQDataError(f"{source}: expected a JSON list of FAQ records.")
    if not raw:
        raise FAQDataError(f"{source}: the FAQ list is empty.")

    entries = [FAQEntry.from_dict(item, position=i) for i, item in enumerate(raw, 1)]

    seen_ids: Dict[str, int] = {}
    duplicate_ids: List[str] = []
    for entry in entries:
        if entry.faq_id in seen_ids and entry.faq_id not in duplicate_ids:
            duplicate_ids.append(entry.faq_id)
        seen_ids[entry.faq_id] = seen_ids.get(entry.faq_id, 0) + 1
    if duplicate_ids:
        raise FAQDataError(
            f"{source}: duplicate FAQ ID(s): {', '.join(sorted(duplicate_ids))}."
        )

    seen_questions: Dict[str, str] = {}
    for entry in entries:
        key = entry.question.casefold()
        if key in seen_questions:
            raise FAQDataError(
                f"{source}: {entry.faq_id} repeats the question of "
                f"{seen_questions[key]}."
            )
        seen_questions[key] = entry.faq_id
    return entries


def load_faq_entries(path: Union[str, Path]) -> List[FAQEntry]:
    """Load and validate the FAQ JSON file, raising ``FAQDataError`` on any problem."""
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise FAQDataError(f"FAQ file not found: {file_path}") from None
    except json.JSONDecodeError as exc:
        raise FAQDataError(
            f"FAQ file is not valid JSON ({file_path.name}, line {exc.lineno}, "
            f"column {exc.colno}): {exc.msg}"
        ) from None
    except UnicodeDecodeError:
        raise FAQDataError(f"FAQ file is not valid UTF-8 text: {file_path.name}") from None
    except OSError as exc:
        raise FAQDataError(
            f"Could not read FAQ file {file_path}: {exc.strerror or exc}"
        ) from None
    return parse_faq_records(raw, source=file_path.name)
