"""Core question-processing package for the College Enquiry Assistant.

This package contains no Flask code, so the retrieval logic can be used and
evaluated without a browser or web server.
"""

from .enquiry_assistant import EnquiryAssistant
from .faq_entry import FAQDataError, FAQEntry, load_faq_entries
from .query_processor import QueryProcessor, normalize_text

__all__ = [
    "EnquiryAssistant",
    "FAQDataError",
    "FAQEntry",
    "QueryProcessor",
    "load_faq_entries",
    "normalize_text",
]
