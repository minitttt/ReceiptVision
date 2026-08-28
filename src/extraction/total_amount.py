"""
extraction/total_amount.py — Total amount extraction.
"""

import re
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)

_PRICE_RE = re.compile(config.PRICE_PATTERN, re.IGNORECASE)
_TOTAL_KEYWORD_RE = re.compile(
    r"(?<![a-zA-Z])(" + "|".join(re.escape(kw) for kw in config.TOTAL_KEYWORDS) + r")(?![a-zA-Z])",
    re.IGNORECASE,
)
_CASH_KEYWORD_RE = re.compile(
    r"(?<![a-zA-Z])(cash|change|tender|paid|amount paid|received|visa|master|card|debit|credit)(?![a-zA-Z])",
    re.IGNORECASE,
)
_SUBTOTAL_KEYWORD_RE = re.compile(
    r"(?<![a-zA-Z])(sub.?total|sub|tax|gst|vat|service charge|discount|rounding)(?![a-zA-Z])",
    re.IGNORECASE,
)
_PRODUCT_TOTAL_RE = re.compile(
    r"\b[a-z]{2,}\s+total\s+[a-z]{2,}\b",
    re.IGNORECASE,
)


def _parse_price(text: str) -> Optional[float]:
    """
    Extract and normalise the first currency amount from a string.

    Handles both decimal-comma (European: 1,99) and thousands-separator-comma
    (Asian/US: 74,000 or 1,234.56) formats.

    Rules:
    - If the string matches X,YYY or X,YYY,ZZZ (comma before 3+ digits with
      no trailing decimal part), treat commas as thousands separators.
    - If the string ends in ,XX (exactly 2 digits after a single comma),
      treat that comma as a decimal separator.
    - Otherwise fall back to stripping all commas and parsing as float.
    """
    m = _PRICE_RE.search(text)
    if not m:
        return None
    raw = m.group(1).replace(" ", "")

    thousands_re = re.compile(r'^\d{1,3}(,\d{3})+(\.\d+)?$')
    euro_decimal_re = re.compile(r'^\d+,\d{2}$')

    try:
        if thousands_re.match(raw):
            normalized = raw.replace(',', '')
        elif euro_decimal_re.match(raw):
            normalized = raw.replace(',', '.')
        else:
            normalized = raw.replace(',', '')
        return float(normalized)
    except ValueError:
        return None


def _normalise_price_str(text: str) -> Optional[str]:
    """Return a clean price string (period decimal) or None."""
    val = _parse_price(text)
    return f"{val:.2f}" if val is not None else None


def _vertical_center(token: dict) -> float:
    """Return the vertical center of an OCR token's bounding box."""
    return sum(point[1] for point in token["bbox"]) / len(token["bbox"])


def _is_embedded_identifier(text: str) -> bool:
    """Reject amounts extracted from barcodes, receipt IDs, or serials."""
    match = _PRICE_RE.search(text)
    if not match:
        return False
    start, end = match.span(1)
    return (
        (start > 0 and text[start - 1].isalnum())
        or (end < len(text) and text[end].isalnum())
    )


def extract_total(tokens: list[dict]) -> tuple[Optional[str], dict]:
    """
    Extract the grand total amount from OCR tokens.

    Strategy:
    1. Scan tokens bottom-up (totals appear near the end of receipts).
    2. Score each candidate by:
       - Keyword adjacency: token text or adjacent token contains a total keyword.
       - Position: lower position in the receipt → higher priority.
       - Magnitude: larger amounts are preferred over subtotals.
    3. Return the highest-scoring candidate as ``value``, with any alternative
       candidates exposed for transparency.

    Args:
        tokens: Sorted OCR token dicts (top-to-bottom order).

    Returns:
        Tuple of:
            - value: Price string (e.g. ``"24.99"``), or None.
            - evidence: Dict with ``ocr_confidence``, ``heuristic_score``,
              ``keyword_match``, ``position_from_bottom``, ``alternative_candidates``.
    """
    candidates = []
    total_tokens = len(tokens)

    for idx, token in enumerate(reversed(tokens)):
        position_from_bottom = idx
        text = token["text"]

        price_val = _parse_price(text)
        if price_val is None or _is_embedded_identifier(text):
            continue

        current_y = _vertical_center(token)
        line_tolerance = max(20.0, token.get("bbox_height", 0.0) * 0.75)
        window_tokens = [
            candidate for candidate in tokens
            if abs(_vertical_center(candidate) - current_y) <= line_tolerance
        ]
        window_text = " ".join(t["text"] for t in window_tokens)

        cash_match    = bool(_CASH_KEYWORD_RE.search(window_text))
        subtotal_match = bool(_SUBTOTAL_KEYWORD_RE.search(window_text))
        keyword_match = (
            bool(_TOTAL_KEYWORD_RE.search(window_text))
            and not subtotal_match
            and not _PRODUCT_TOTAL_RE.search(window_text)
        )

        if cash_match and not keyword_match:
            continue

        position_score = max(0.0, 1.0 - (position_from_bottom / max(total_tokens, 1) * 3))
        keyword_score = 1.0 if keyword_match else (0.2 if subtotal_match else 0.4)
        magnitude_score = min(1.0, price_val / 1000.0)
        plausibility_penalty = 0.3 if (price_val > 9999 and not keyword_match) else 0.0

        heuristic = max(0.0,
            0.5 * keyword_score + 0.35 * position_score + 0.15 * magnitude_score
            - plausibility_penalty
        )

        candidates.append({
            "value": f"{price_val:.2f}",
            "price_float": price_val,
            "ocr_confidence": token["confidence"],
            "heuristic_score": round(heuristic, 4),
            "keyword_match": keyword_match,
            "position_from_bottom": position_from_bottom,
            "raw_text": text,
        })

    keyword_candidates = [candidate for candidate in candidates if candidate["keyword_match"]]
    if not keyword_candidates:
        logger.info("No total amount found in OCR output.")
        return None, {
            "ocr_confidence": 0.0,
            "heuristic_score": 0.0,
            "keyword_match": False,
            "position_from_bottom": -1,
            "alternative_candidates": [
                {k: v for k, v in candidate.items() if k != "price_float"}
                for candidate in candidates[:3]
            ],
        }

    candidates.sort(
        key=lambda c: (c["keyword_match"], c["heuristic_score"], c["price_float"]),
        reverse=True,
    )

    best = candidates[0]
    alternatives = [
        {k: v for k, v in c.items() if k != "price_float"}
        for c in candidates[1:4]
    ]

    logger.debug("Total amount: %s (keyword=%s, heuristic=%.3f)",
                 best["value"], best["keyword_match"], best["heuristic_score"])

    return best["value"], {
        "ocr_confidence": best["ocr_confidence"],
        "heuristic_score": best["heuristic_score"],
        "keyword_match": best["keyword_match"],
        "position_from_bottom": best["position_from_bottom"],
        "alternative_candidates": alternatives,
    }
