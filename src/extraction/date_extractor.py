"""
extraction/date_extractor.py — Date of transaction extraction.
"""

import re
import logging
from typing import Optional

from dateutil import parser as dateutil_parser
from dateutil.parser import ParserError

import config

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

logger = logging.getLogger(__name__)


def _try_parse_date(text: str, yearfirst: bool = False) -> Optional[str]:
    """
    Attempt to parse a date string using dateutil strict (non-fuzzy) parsing.

    Returns an ISO 8601 date string (YYYY-MM-DD) on success, or None if the
    text cannot be unambiguously parsed as a date.

    Args:
        text: The raw date string to parse.
        yearfirst: If True, interpret the first numeric component as the year
                   (correct for ISO 8601 / YYYY-MM-DD patterns).
    """
    try:
        dt = dateutil_parser.parse(text, fuzzy=False, yearfirst=yearfirst, dayfirst=False)
        if not (1990 <= dt.year <= 2099):
            return None
        return dt.strftime("%Y-%m-%d")
    except (ParserError, ValueError, OverflowError):
        return None


def extract_date(tokens: list[dict]) -> tuple[Optional[str], dict]:
    """
    Extract the transaction date from OCR tokens.

    Strategy:
    1. For each token, run all regex patterns from ``config.DATE_PATTERNS``.
    2. Validate matched strings via ``dateutil.parser.parse`` to reject false
       positives (e.g. phone numbers / item codes that happen to match digits).
    3. Return the first confident match, normalised to ISO 8601 (YYYY-MM-DD).

    Args:
        tokens: Sorted list of OCR token dicts.

    Returns:
        Tuple of:
            - value: ISO 8601 date string, or None if not found.
            - evidence: Dict with ``ocr_confidence``, ``pattern_validity``,
              ``matched_raw``, ``pattern_index``.
    """
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in config.DATE_PATTERNS]

    candidate_texts = []
    for i, token in enumerate(tokens):
        candidate_texts.append((token["text"], [token]))
        if i < len(tokens) - 1:
            combined = token["text"] + " " + tokens[i + 1]["text"]
            candidate_texts.append((combined, [token, tokens[i + 1]]))

    for text, source_tokens in candidate_texts:
        for pat_idx, pattern in enumerate(compiled_patterns):
            match = pattern.search(text)
            if match:
                raw_match = match.group(0).strip()
                yearfirst = pat_idx == 0
                parsed = _try_parse_date(raw_match, yearfirst=yearfirst)
                if parsed:
                    avg_conf = sum(t["confidence"] for t in source_tokens) / len(source_tokens)
                    logger.debug("Date found: '%s' → '%s' (pattern=%d, conf=%.3f)",
                                 raw_match, parsed, pat_idx, avg_conf)
                    return parsed, {
                        "ocr_confidence": avg_conf,
                        "pattern_validity": 1.0,
                        "matched_raw": raw_match,
                        "pattern_index": pat_idx,
                    }

    for token in tokens:
        if not _YEAR_RE.search(token["text"]):
            continue
        try:
            dt = dateutil_parser.parse(token["text"], fuzzy=True, dayfirst=False)
            if 1990 <= dt.year <= 2099:
                parsed = dt.strftime("%Y-%m-%d")
                logger.debug("Date found via fuzzy fallback: '%s' → '%s'", token["text"], parsed)
                return parsed, {
                    "ocr_confidence": token["confidence"],
                    "pattern_validity": 0.5,
                    "matched_raw": token["text"],
                    "pattern_index": -1,
                }
        except (ParserError, ValueError, OverflowError):
            continue

    logger.info("No date found in OCR output.")
    return None, {"ocr_confidence": 0.0, "pattern_validity": 0.0, "matched_raw": None, "pattern_index": -1}
