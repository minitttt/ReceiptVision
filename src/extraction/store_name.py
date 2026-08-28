"""
extraction/store_name.py — Store / vendor name extraction.
"""

import re
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


def _bbox_height(bbox: list) -> float:
    """Return pixel height of a bounding box."""
    ys = [pt[1] for pt in bbox]
    return max(ys) - min(ys)


def extract_store_name(
    tokens: list[dict],
    top_n: int = None,
) -> tuple[Optional[str], dict]:
    """
    Extract the store / vendor name from OCR tokens.

    Heuristic: the store name is typically printed in a larger font at the
    very top of the receipt (first 1–3 non-empty lines).  We select the
    candidate with the tallest bounding box (largest font size proxy).

    If the primary heuristic fails (no tokens in the top region), we fall back
    to the single highest-confidence token in the top third of the image.

    Args:
        tokens: Sorted list of OCR token dicts ``{text, confidence, bbox, bbox_height}``.
        top_n: Number of leading tokens to consider. Defaults to ``config.STORE_TOP_LINES``.

    Returns:
        Tuple of:
            - value: Extracted store name string, or None.
            - evidence: Dict with ``ocr_confidence``, ``heuristic_score``, ``source``.
    """
    top_n = top_n or config.STORE_TOP_LINES

    meaningful = [t for t in tokens if t["text"].strip()]

    if not meaningful:
        logger.warning("No meaningful OCR tokens — cannot extract store name.")
        return None, {"ocr_confidence": 0.0, "heuristic_score": 0.0, "source": "none"}

    def _is_viable_store_token(t: dict) -> bool:
        text = t["text"].strip()
        if t["confidence"] < 0.25:
            return False
        if re.match(r"^[\d\s\-\#\:\.\/\,]+$", text):
            return False
        if re.match(r"^[A-Za-z0-9\-]+$", text) and any(c.isdigit() for c in text) and len(text) < 15:
            return False
        if len(text) <= 2:
            return False
        skip_patterns = re.compile(
            r"\b(receipt|rcpt|cashier|operator|date|time|check|pax|pos|table|cash|change|thank|win|chance|survey|feedback|visit|sweepstakes|see|www|http|\.com)\b",
            re.IGNORECASE
        )
        if skip_patterns.search(text):
            return False
        return True

    top_candidates = meaningful[:top_n]
    viable = [t for t in top_candidates if _is_viable_store_token(t)]

    if not viable:
        viable = top_candidates

    def _score(t: dict) -> float:
        h = t.get("bbox_height", _bbox_height(t["bbox"]))
        return h * t["confidence"]

    best = max(viable, key=_score)

    position = top_candidates.index(best) if best in top_candidates else 0
    heuristic_score = 1.0 - (position * 0.1)

    logger.debug("Store name extracted: '%s' (pos=%d, bbox_h=%.1f, conf=%.3f)",
                 best["text"], position, best.get("bbox_height", 0), best["confidence"])

    return best["text"].strip(), {
        "ocr_confidence": best["confidence"],
        "heuristic_score": max(0.0, heuristic_score),
        "source": "top_n_largest_bbox",
    }

