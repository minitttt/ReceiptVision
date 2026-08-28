"""
preprocessing/denoise.py — Noise and blur handling utilities.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def denoise(img: np.ndarray) -> np.ndarray:
    """
    Apply Non-Local Means denoising to a BGR image.

    Non-local means works well for JPEG compression artifacts and
    thermal-print grain that commonly appear in receipt images.

    Args:
        img: BGR image as numpy array.

    Returns:
        Denoised BGR image.
    """
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def compute_blur_score(img: np.ndarray) -> float:
    """
    Compute the Laplacian variance as an objective blur metric.

    Higher variance = sharper image. Use as a proxy for expected OCR quality.

    Args:
        img: BGR or grayscale image.

    Returns:
        Laplacian variance (float). Values below ~100 indicate noticeable blur.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sharpen_if_blurry(img: np.ndarray, threshold: float = 100.0) -> tuple[np.ndarray, float]:
    """
    Sharpen the image if its blur score falls below ``threshold``.

    Uses a standard unsharp-mask sharpening kernel. The blur score is
    always returned regardless of whether sharpening was applied, so
    callers can log it as an image-quality metric.

    Args:
        img: BGR image as numpy array.
        threshold: Laplacian variance below which sharpening is applied.

    Returns:
        Tuple of (processed image, blur_score).
    """
    blur_score = compute_blur_score(img)
    if blur_score < threshold:
        logger.debug("Image is blurry (score=%.2f < %.2f); applying sharpening.", blur_score, threshold)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        img = cv2.filter2D(img, -1, kernel)
    else:
        logger.debug("Image sharpness OK (score=%.2f).", blur_score)
    return img, blur_score
