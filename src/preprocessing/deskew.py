"""
preprocessing/deskew.py — Skew / rotation correction utilities.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

EXTREME_ANGLE_THRESHOLD = 30.0


def deskew(img: np.ndarray, extreme_threshold: float = EXTREME_ANGLE_THRESHOLD) -> tuple[np.ndarray, float, bool]:
    """
    Detect and correct rotational skew in a receipt image.

    Uses Otsu binarisation + ``cv2.minAreaRect`` on foreground pixel
    coordinates to estimate the dominant text-line angle.  Falls back to
    returning the original image unchanged if the algorithm cannot find a
    reliable angle (e.g. image has very sparse text).

    Args:
        img: BGR image as numpy array.
        extreme_threshold: Angles beyond this value are flagged as extreme.

    Returns:
        Tuple of (deskewed image, detected_angle_degrees, is_extreme_skew).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 5:
        logger.warning("Too few foreground pixels for reliable deskew; returning original.")
        return img, 0.0, False

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    is_extreme = abs(angle) > extreme_threshold
    if is_extreme:
        logger.warning("Extreme skew detected: %.2f°. Flagging image.", angle)

    logger.debug("Detected skew angle: %.2f°", angle)

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return deskewed, angle, is_extreme
