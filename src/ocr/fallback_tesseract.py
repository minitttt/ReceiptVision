"""
ocr/fallback_tesseract.py — Tesseract fallback OCR engine.

Used when EasyOCR mean confidence falls below config.OCR_FALLBACK_THRESHOLD.
Normalises Tesseract output to the same token schema as the EasyOCR engine
so downstream code is engine-agnostic.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_TESSERACT_AVAILABLE = None


def _check_tesseract() -> bool:
    """Return True if pytesseract and the Tesseract binary are available."""
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is not None:
        return _TESSERACT_AVAILABLE
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _TESSERACT_AVAILABLE = True
        logger.info("Tesseract binary found — fallback available.")
    except Exception as exc:
        _TESSERACT_AVAILABLE = False
        logger.warning("Tesseract not available (%s). Fallback disabled.", exc)
    return _TESSERACT_AVAILABLE


def run_tesseract(
    image_input,
    receipt_id: str,
) -> tuple[list[dict], float]:
    """
    Run Tesseract OCR and return results in the same schema as EasyOCR.

    Uses ``pytesseract.image_to_data`` which provides per-word confidence
    scores so we can compare directly with EasyOCR output.

    Args:
        image_input: File path (str/Path) or numpy BGR/gray array.
        receipt_id: Used for log messages.

    Returns:
        Tuple of:
            - tokens: List of ``{text, confidence, bbox, bbox_height}``.
            - mean_confidence: Mean confidence (0.0 if Tesseract unavailable).
    """
    if not _check_tesseract():
        logger.warning("[%s] Tesseract unavailable — returning empty token list.", receipt_id)
        return [], 0.0

    import pytesseract
    from PIL import Image

    logger.info("[%s] Running Tesseract fallback…", receipt_id)

    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(str(image_input))
    elif isinstance(image_input, np.ndarray):
        import cv2
        rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
    else:
        pil_img = image_input

    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

    tokens = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf_raw = int(data["conf"][i])
        if not text or conf_raw < 0:
            continue

        conf = conf_raw / 100.0

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

        tokens.append({
            "text": text,
            "confidence": conf,
            "bbox": bbox,
            "bbox_height": float(h),
        })

    tokens = sorted(tokens, key=lambda t: (t["bbox"][0][1], t["bbox"][0][0]))

    mean_conf = sum(t["confidence"] for t in tokens) / len(tokens) if tokens else 0.0
    logger.info("[%s] Tesseract: %d tokens, mean_conf=%.3f", receipt_id, len(tokens), mean_conf)
    return tokens, mean_conf
