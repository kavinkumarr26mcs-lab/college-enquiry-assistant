"""QueryProcessor: TF-IDF + cosine-similarity retrieval over the FAQ dataset.

This module has no Flask dependency, so it can be used and evaluated directly.

Pipeline for every question
---------------------------
1. Validate the input (type, empty, maximum length).
2. Normalise: lower-case, strip punctuation, drop stop words, light plural
   stemming, and a small editable domain lexicon (``DEFAULT_SYNONYMS``).
3. Vectorise with TF-IDF (unigrams + bigrams) fitted on the FAQ questions and
   their approved paraphrases.
4. Cosine similarity against every FAQ text; each FAQ keeps its best score.
5. Scale by an out-of-vocabulary penalty: ``(known / all terms) ** exponent``.
6. Return the best FAQ only if the score reaches the threshold and clearly beats
   the runner-up; otherwise return the fallback.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    DEFAULT_COVERAGE_EXPONENT,
    DEFAULT_FAQ_PATH,
    DEFAULT_MIN_MARGIN,
    DEFAULT_SIMILARITY_THRESHOLD,
    FALLBACK_MESSAGE,
    MAX_QUESTION_LENGTH,
    validate_unit_interval,
)
from .faq_entry import FAQDataError, FAQEntry, load_faq_entries

STOP_WORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been
    before being below between both but by can cannot could did do does doing
    down during each few for from further get give had has have having he her
    here hers him his how i if in into is it its just kindly let like may me
    might more most must my no nor not of off on once only or other our ours
    out over own please same shall she should so some such than that the their
    theirs them then there these they this those through to too under until up
    us very want was we were what when where which while who whom whose why
    will with would you your yours yes tell know information details
    available availability offer offered offers provide provided
    need needs needed require required requires requirement requirements
    """.split()
)

# Multi-word expressions rewritten before tokenising (matched on whole words).
DEFAULT_PHRASES: Dict[str, str] = {
    "wi fi": "wifi",
    "admit card": "hall ticket",
    "co education": "coeducation",
    "re evaluation": "revaluation",
    "bona fide": "bonafide",
    "financial aid": "scholarship",
    "tc": "transfer certificate",
}

# Single-word domain lexicon. Keys may be raw or already-stemmed forms.
DEFAULT_SYNONYMS: Dict[str, str] = {
    "programme": "course", "program": "course",
    "applying": "apply", "applied": "apply", "register": "apply",
    "registration": "apply", "enrol": "apply", "enroll": "apply",
    "enrolment": "apply", "enrollment": "apply",
    "payment": "pay", "paying": "pay", "paid": "pay",
    "borrowing": "borrow", "borrowed": "borrow",
    "timing": "time", "hour": "time",
    "accommodation": "hostel", "dormitory": "hostel", "residence": "hostel",
    "tuition": "fee", "cost": "fee", "price": "fee", "charge": "fee",
    "bus": "transport", "buse": "transport", "conveyance": "transport",
    "internet": "wifi",
    "cafeteria": "canteen", "mess": "canteen", "food": "canteen",
    "complaint": "grievance",
    "counselling": "counsel", "counseling": "counsel",
    "counsellor": "counsel", "counselor": "counsel",
    "bursary": "scholarship", "stipend": "scholarship", "aid": "scholarship",
    "job": "placement", "recruitment": "placement", "recruiter": "placement",
    "career": "placement",
    "recheck": "revaluation", "rechecking": "revaluation",
    "reevaluation": "revaluation",
    "located": "location", "address": "location", "situated": "location",
    "reach": "location",
    "phone": "contact", "telephone": "contact", "email": "contact",
    "call": "contact",
    "lab": "laboratory",
    "ug": "undergraduate", "bachelor": "undergraduate",
    "pg": "postgraduate", "master": "postgraduate",
    "identity": "id",
    "installment": "instalment",
    "reimbursement": "refund",
    "coed": "coeducation", "coeducational": "coeducation",
    "examination": "exam",
    "game": "sport",
    "doctor": "medical", "clinic": "medical", "infirmary": "medical",
    "nurse": "medical",
    "tour": "visit",
}

_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


def _stem(token: str) -> str:
    """Very light plural stemming (courses->course, libraries->library)."""
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    if token.endswith(("ss", "us", "is")):
        return token
    if token.endswith("s"):
        return token[:-1]
    return token


def normalize_text(
    text: Any,
    synonyms: Optional[Mapping[str, str]] = None,
    phrases: Optional[Mapping[str, str]] = None,
    stop_words: Optional[frozenset] = None,
) -> str:
    """Return the searchable form of ``text`` (empty string if nothing remains)."""
    if not isinstance(text, str):
        return ""
    synonyms = DEFAULT_SYNONYMS if synonyms is None else synonyms
    phrases = DEFAULT_PHRASES if phrases is None else phrases
    stop_words = STOP_WORDS if stop_words is None else stop_words

    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = _NON_WORD.sub(" ", folded).strip()
    for phrase, replacement in phrases.items():
        folded = re.sub(rf"\b{re.escape(phrase)}\b", replacement, folded)

    tokens: List[str] = []
    for token in folded.split():
        if len(token) < 2 or token in stop_words:
            continue
        token = synonyms.get(token, token)
        stemmed = _stem(token)
        tokens.append(synonyms.get(stemmed, stemmed))
    return " ".join(tokens)


class QueryProcessor:
    """Finds the FAQ that best answers a question, or reports that none does."""

    def __init__(
        self,
        faq_path: Union[str, Path, None] = None,
        entries: Optional[Sequence[FAQEntry]] = None,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        coverage_exponent: float = DEFAULT_COVERAGE_EXPONENT,
        min_margin: float = DEFAULT_MIN_MARGIN,
        max_question_length: int = MAX_QUESTION_LENGTH,
        synonyms: Optional[Mapping[str, str]] = None,
        phrases: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._threshold = validate_unit_interval("similarity threshold", threshold)
        self._min_margin = validate_unit_interval("minimum margin", min_margin)
        if isinstance(coverage_exponent, bool) or coverage_exponent < 0:
            raise ValueError("coverage_exponent must be zero or positive.")
        self._coverage_exponent = float(coverage_exponent)
        if max_question_length < 1:
            raise ValueError("max_question_length must be at least 1.")
        self._max_question_length = int(max_question_length)
        self._synonyms = DEFAULT_SYNONYMS if synonyms is None else dict(synonyms)
        self._phrases = DEFAULT_PHRASES if phrases is None else dict(phrases)

        if entries is not None:
            loaded = list(entries)
            if not loaded:
                raise FAQDataError("At least one FAQ entry is required.")
            if not all(isinstance(item, FAQEntry) for item in loaded):
                raise FAQDataError("entries must be FAQEntry objects.")
            self._entries: List[FAQEntry] = loaded
        else:
            self._entries = load_faq_entries(faq_path or DEFAULT_FAQ_PATH)
        self._build_index()

    # ------------------------------------------------------------------ properties
    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = validate_unit_interval("similarity threshold", value)

    @property
    def entries(self) -> List[FAQEntry]:
        return list(self._entries)

    @property
    def faq_ids(self) -> List[str]:
        return [entry.faq_id for entry in self._entries]

    # ------------------------------------------------------------------ indexing
    def normalize(self, text: Any) -> str:
        return normalize_text(text, self._synonyms, self._phrases)

    def _build_index(self) -> None:
        documents: List[str] = []
        owners: List[int] = []
        for position, entry in enumerate(self._entries):
            for text in entry.texts():
                normalized = self.normalize(text)
                if normalized:
                    documents.append(normalized)
                    owners.append(position)
                elif text == entry.question:
                    raise FAQDataError(
                        f"{entry.faq_id}: the question has no searchable words."
                    )
        self._doc_faq_index = np.asarray(owners, dtype=int)
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=False,
            norm="l2",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self._matrix = self._vectorizer.fit_transform(documents)

    # ------------------------------------------------------------------ scoring
    def _score_faqs(self, text: str) -> Optional[np.ndarray]:
        """Adjusted similarity of ``text`` to every FAQ, or None if no known words."""
        tokens = self.normalize(text).split()
        vocabulary = self._vectorizer.vocabulary_
        known = sum(1 for token in tokens if token in vocabulary)
        if not tokens or known == 0:
            return None
        query_vector = self._vectorizer.transform([" ".join(tokens)])
        similarities = cosine_similarity(query_vector, self._matrix).ravel()
        per_faq = np.zeros(len(self._entries))
        np.maximum.at(per_faq, self._doc_faq_index, similarities)
        coverage = known / len(tokens)
        return per_faq * (coverage ** self._coverage_exponent)

    def _fallback(self, reason: str, similarity: float = 0.0) -> Dict[str, Any]:
        return {
            "matched": False,
            "faq_id": None,
            "question": None,
            "answer": FALLBACK_MESSAGE,
            "category": None,
            "similarity": round(float(similarity), 4),
            "reason": reason,
        }

    def process(self, question: Any) -> Dict[str, Any]:
        """Answer ``question`` from the FAQ data. Never raises for bad input."""
        if not isinstance(question, str):
            return self._fallback("invalid_input")
        cleaned = question.strip()
        if not cleaned:
            return self._fallback("empty_input")
        if len(cleaned) > self._max_question_length:
            return self._fallback("input_too_long")

        scores = self._score_faqs(cleaned)
        if scores is None:
            return self._fallback("no_known_terms")

        order = np.argsort(-scores, kind="stable")
        best_index = int(order[0])
        best = float(scores[best_index])
        runner_up = float(scores[order[1]]) if len(order) > 1 else 0.0

        if best < self._threshold:
            return self._fallback("below_threshold", best)
        if best - runner_up < self._min_margin:
            return self._fallback("ambiguous_match", best)

        entry = self._entries[best_index]
        return {
            "matched": True,
            "faq_id": entry.faq_id,
            "question": entry.question,
            "answer": entry.answer,
            "category": entry.category,
            "similarity": round(best, 4),
            "reason": "reliable_match",
        }

    def top_matches(self, question: Any, k: int = 3) -> List[Dict[str, Any]]:
        """The ``k`` closest FAQs with their scores (for debugging and tuning)."""
        if not isinstance(question, str) or not question.strip():
            return []
        scores = self._score_faqs(question.strip()[: self._max_question_length])
        if scores is None:
            return []
        order = np.argsort(-scores, kind="stable")[: max(int(k), 0)]
        return [
            {
                "faq_id": self._entries[i].faq_id,
                "question": self._entries[i].question,
                "similarity": round(float(scores[i]), 4),
            }
            for i in order
        ]
