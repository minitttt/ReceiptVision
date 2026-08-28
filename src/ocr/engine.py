"""
ocr/engine.py — EasyOCR wrapper with result caching and sorting.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

_reader = None


def _get_reader():
    """Lazy-initialise the EasyOCR Reader singleton."""
    global _reader
    if _reader is None:
        try:
            import easyocr
            logger.info("Initialising EasyOCR reader (language=%s, gpu=%s)…",
                        config.OCR_LANGUAGE, config.OCR_GPU)
            _reader = easyocr.Reader(config.OCR_LANGUAGE, gpu=config.OCR_GPU)
        except ImportError:
            logger.error("EasyOCR not installed. Run: pip install easyocr")
            raise
    return _reader


def _sort_ocr_results(results: list[dict]) -> list[dict]:
    """
    Sort OCR token results in reading order: top-to-bottom, then left-to-right.

    Each bbox is a list of four [x, y] corner points.  We use the mean y of
    the top two corners as the primary sort key and the mean x as secondary.
    """
    def sort_key(token):
        bbox = token["bbox"]
        top_y = (bbox[0][1] + bbox[1][1]) / 2
        left_x = (bbox[0][0] + bbox[3][0]) / 2
        return (top_y, left_x)

    return sorted(results, key=sort_key)


def _bbox_height(bbox: list) -> float:
    """Return the pixel height of an OCR bounding box."""
    ys = [pt[1] for pt in bbox]
    return max(ys) - min(ys)


def run_easyocr(
    image_input,
    receipt_id: str,
    cache_dir: Optional[str] = None,
) -> tuple[list[dict], float]:
    """
    Run EasyOCR on an image and return sorted token results + mean confidence.

    Args:
        image_input: File path (str/Path) or numpy BGR array.
        receipt_id: Used for cache file naming.
        cache_dir: Directory to cache raw OCR JSON output.
                   Defaults to ``config.OCR_CACHE_DIR``.

    Returns:
        Tuple of:
            - tokens: List of dicts ``{text, confidence, bbox, bbox_height}``.
            - mean_confidence: Mean confidence across all tokens (0.0 if empty).
    """
    cache_dir = cache_dir or config.OCR_CACHE_DIR
    cache_path = Path(cache_dir) / f"{receipt_id}_easyocr.json"

    if cache_path.exists():
        logger.info("[%s] Loading cached EasyOCR results from %s", receipt_id, cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            tokens = json.load(f)
        mean_conf = _mean_confidence(tokens)
        return tokens, mean_conf

    reader = _get_reader()
    logger.info("[%s] Running EasyOCR…", receipt_id)

    if isinstance(image_input, (str, Path)):
        raw_results = reader.readtext(str(image_input))
    else:
        raw_results = reader.readtext(image_input)

    tokens = [
        {
            "text": text,
            "confidence": float(conf),
            "bbox": [[int(pt[0]), int(pt[1])] for pt in bbox],
            "bbox_height": float(_bbox_height(bbox)),
        }
        for bbox, text, conf in raw_results
    ]
    tokens = _sort_ocr_results(tokens)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(tokens, f, indent=2)
    logger.info("[%s] EasyOCR: %d tokens, mean_conf=%.3f. Cached to %s",
                receipt_id, len(tokens), _mean_confidence(tokens), cache_path)

    mean_conf = _mean_confidence(tokens)
    return tokens, mean_conf


def _mean_confidence(tokens: list[dict]) -> float:
    """Compute mean OCR confidence across a token list."""
    if not tokens:
        return 0.0
    return sum(t["confidence"] for t in tokens) / len(tokens)
