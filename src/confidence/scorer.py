"""
confidence/scorer.py — Field-level confidence scoring and reliability flagging.
"""

import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


def score_field(
    ocr_confidence: float,
    pattern_validity: float,
    heuristic_score: float,
) -> float:
    """
    Compute a composite confidence score for a single extracted field.

    Formula (weights configured in config.py):
        score = 0.50 * ocr_confidence
              + 0.30 * pattern_validity
              + 0.20 * heuristic_score

    All inputs are expected in [0.0, 1.0].

    Args:
        ocr_confidence: Mean OCR confidence of the tokens composing the field.
        pattern_validity: 1.0 = clean regex/dateutil match;
                          0.5 = fuzzy/partial match;
                          0.0 = no match found.
        heuristic_score: Positional / keyword-adjacency heuristic score.

    Returns:
        Composite confidence score clamped to [0.0, 1.0].
    """
    raw = (
        config.CONF_WEIGHT_OCR * ocr_confidence
        + config.CONF_WEIGHT_PATTERN * pattern_validity
        + config.CONF_WEIGHT_HEURISTIC * heuristic_score
    )
    return round(min(1.0, max(0.0, raw)), 3)


def build_field(
    value,
    ocr_confidence: float,
    pattern_validity: float,
    heuristic_score: float,
    flag_threshold: float = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Assemble the full confidence-scored field dict for the output schema.

    Adds ``flagged: true`` and a ``reason`` string when confidence is below
    ``flag_threshold``.

    Args:
        value: Extracted value (str, float, or None).
        ocr_confidence: Raw OCR confidence for this field.
        pattern_validity: Pattern match quality (0–1).
        heuristic_score: Heuristic quality (0–1).
        flag_threshold: Override ``config.CONF_FLAG_THRESHOLD`` if needed.
        extra: Optional extra keys to merge into the output dict.

    Returns:
        Dict with ``value``, ``confidence``, and optionally ``flagged``,
        ``reason``, and any keys from ``extra``.
    """
    flag_threshold = flag_threshold if flag_threshold is not None else config.CONF_FLAG_THRESHOLD
    confidence = score_field(ocr_confidence, pattern_validity, heuristic_score)

    field: dict = {"value": value, "confidence": confidence}

    if value is None:
        field["flagged"] = True
        field["reason"] = "field not found in receipt"
    elif confidence < flag_threshold:
        field["flagged"] = True
        reasons = []
        if ocr_confidence < 0.5:
            reasons.append("low OCR confidence")
        if pattern_validity < 0.5:
            reasons.append("pattern mismatch")
        if heuristic_score < 0.3:
            reasons.append("weak positional heuristic")
        field["reason"] = "; ".join(reasons) if reasons else "confidence below threshold"

    if extra:
        field.update(extra)

    return field


def score_items(raw_items: list[dict]) -> list[dict]:
    """
    Build confidence-scored item dicts from raw extraction output.

    For items, the pattern_validity is 1.0 (the regex matched, otherwise the
    item wouldn't have been extracted) and the heuristic is set to 0.8 as a
    reasonable default for itemised line detection.

    Args:
        raw_items: List of dicts from ``extract_items``.

    Returns:
        List of scored item dicts conforming to the output schema.
    """
    scored = []
    for item in raw_items:
        field = build_field(
            value=item.get("price"),
            ocr_confidence=item.get("confidence", 0.5),
            pattern_validity=1.0,
            heuristic_score=0.8,
        )
        scored.append({
            "name": item.get("name", ""),
            "price": field["value"],
            "confidence": field["confidence"],
            **({"flagged": field["flagged"], "reason": field["reason"]}
               if "flagged" in field else {}),
        })
    return scored
