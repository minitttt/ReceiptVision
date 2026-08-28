"""
preprocessing/contrast.py — Lighting, contrast, and binarisation utilities.
"""

import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def fix_lighting(img: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
    luminance channel of the image in LAB colour space.

    This approach improves local contrast without washing out well-lit regions,
    making faded thermal-print receipts significantly more legible for OCR.

    Args:
        img: BGR image as numpy array.
        clip_limit: CLAHE clip limit (higher → more aggressive).
        tile_grid: Grid size for local histogram equalization.

    Returns:
        Contrast-enhanced BGR image.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def adaptive_binarize(img: np.ndarray, block_size: int = 31, c_value: int = 15) -> np.ndarray:
    """
    Convert to grayscale and apply Gaussian adaptive thresholding.

    Adaptive thresholding handles uneven illumination (shadows, glare) far
    better than a global threshold, producing clean black-on-white text
    suitable for OCR engines.

    Args:
        img: BGR image (or grayscale).
        block_size: Size of the neighbourhood area (must be odd).
        c_value: Constant subtracted from the mean.

    Returns:
        Binary (grayscale) image.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c_value,
    )


def preprocess_pipeline(
    img: np.ndarray,
    receipt_id: str,
    processed_dir: str = "data/processed",
    blur_threshold: float = 100.0,
    clahe_clip: float = 2.0,
    clahe_tile: tuple = (8, 8),
    adaptive_block: int = 31,
    adaptive_c: int = 15,
    skew_extreme: float = 30.0,
    save_debug: bool = True,
) -> dict:
    """
    Run the full preprocessing chain and return all artefacts + metadata.

    Chain order (important — do NOT reorder):
        1. denoise
        2. deskew
        3. CLAHE lighting fix
        4. (saved) adaptive binarization for OCR

    Args:
        img: BGR image as numpy array.
        receipt_id: Unique identifier for the receipt (used for filenames).
        processed_dir: Directory to save debug images.
        blur_threshold: Blur metric threshold for sharpening.
        clahe_clip: CLAHE clip limit.
        clahe_tile: CLAHE tile grid size.
        adaptive_block: Adaptive threshold block size.
        adaptive_c: Adaptive threshold C constant.
        skew_extreme: Angle beyond which skew is flagged as extreme.
        save_debug: Whether to save before/after images to disk.

    Returns:
        Dict with keys:
            - ``processed``: Final preprocessed BGR image (for OCR).
            - ``binarized``: Binarized grayscale image (optional OCR input).
            - ``blur_score``: Laplacian variance of original image.
            - ``skew_angle``: Detected skew angle in degrees.
            - ``extreme_skew``: True if skew was flagged as extreme.
            - ``metadata``: Dict of all computed diagnostics.
    """
    from src.preprocessing.denoise import sharpen_if_blurry, denoise
    from src.preprocessing.deskew import deskew

    metadata: dict = {"receipt_id": receipt_id}

    denoised = denoise(img)

    sharpened, blur_score = sharpen_if_blurry(denoised, threshold=blur_threshold)
    metadata["blur_score"] = round(blur_score, 3)
    logger.info("[%s] Blur score: %.3f", receipt_id, blur_score)

    deskewed, angle, extreme_skew = deskew(sharpened, extreme_threshold=skew_extreme)
    metadata["skew_angle"] = round(angle, 2)
    metadata["extreme_skew"] = extreme_skew

    contrast_fixed = fix_lighting(deskewed, clip_limit=clahe_clip, tile_grid=clahe_tile)

    binarized = adaptive_binarize(contrast_fixed, block_size=adaptive_block, c_value=adaptive_c)

    if save_debug:
        _save_debug_images(img, contrast_fixed, binarized, receipt_id, processed_dir)

    return {
        "processed": contrast_fixed,
        "binarized": binarized,
        "blur_score": blur_score,
        "skew_angle": angle,
        "extreme_skew": extreme_skew,
        "metadata": metadata,
    }


def _save_debug_images(
    original: np.ndarray,
    processed: np.ndarray,
    binarized: np.ndarray,
    receipt_id: str,
    output_dir: str,
) -> None:
    """Save original, processed, and binarized images for visual debugging."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{receipt_id}_original.jpg"), original)
    cv2.imwrite(str(out / f"{receipt_id}_processed.jpg"), processed)
    cv2.imwrite(str(out / f"{receipt_id}_binarized.jpg"), binarized)
    logger.debug("[%s] Debug images saved to %s", receipt_id, output_dir)
