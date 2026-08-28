"""
tests/test_preprocessing.py — Unit tests for image preprocessing utilities.

All tests use synthetic numpy arrays — no real images or OCR needed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestComputeBlurScore:
    def test_perfect_sharp_image(self):
        """A high-frequency checkerboard pattern should have a high blur score."""
        from src.preprocessing.denoise import compute_blur_score
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[::2, ::2] = 255  # checkerboard
        score = compute_blur_score(img)
        assert score > 50.0, f"Expected high sharpness score, got {score}"

    def test_uniform_image_is_blurry(self):
        """A uniform image has zero Laplacian variance — maximally 'blurry'."""
        from src.preprocessing.denoise import compute_blur_score
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        score = compute_blur_score(img)
        assert score == pytest.approx(0.0, abs=1e-3)

    def test_grayscale_input(self):
        """compute_blur_score should accept grayscale (2-D) arrays."""
        from src.preprocessing.denoise import compute_blur_score
        gray = np.random.randint(0, 256, (80, 80), dtype=np.uint8)
        score = compute_blur_score(gray)
        assert isinstance(score, float)


class TestSharpenIfBlurry:
    def test_sharp_image_unchanged_type(self):
        """Sharp images should pass through without raising."""
        from src.preprocessing.denoise import sharpen_if_blurry
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[::2, ::2] = 255
        result, score = sharpen_if_blurry(img, threshold=100.0)
        assert result.shape == img.shape
        assert isinstance(score, float)

    def test_blurry_image_is_sharpened(self):
        """Blurry image should be processed without error; returns same shape."""
        from src.preprocessing.denoise import sharpen_if_blurry
        blurry = np.full((60, 60, 3), 100, dtype=np.uint8)
        result, score = sharpen_if_blurry(blurry, threshold=100.0)
        assert result.shape == blurry.shape
        assert score < 100.0  # uniform image has zero variance


class TestDeskew:
    def test_near_zero_angle_no_crash(self):
        """A well-aligned image should return angle close to 0 without crashing."""
        from src.preprocessing.deskew import deskew
        # Create a simple rectangle of text-like pixels
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        img[80:120, 50:250] = 255  # horizontal band
        result, angle, extreme = deskew(img)
        assert result.shape == img.shape
        assert isinstance(angle, float)
        assert isinstance(extreme, bool)

    def test_sparse_image_returns_original(self):
        """An image with too few pixels should return original gracefully."""
        from src.preprocessing.deskew import deskew
        sparse = np.zeros((100, 100, 3), dtype=np.uint8)
        sparse[50, 50] = 1  # single pixel — too few for minAreaRect
        result, angle, extreme = deskew(sparse)
        assert result.shape == sparse.shape
        assert angle == 0.0


class TestFixLighting:
    def test_output_shape_preserved(self):
        """fix_lighting should return same shape as input."""
        from src.preprocessing.contrast import fix_lighting
        img = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
        result = fix_lighting(img)
        assert result.shape == img.shape

    def test_dark_image_gets_brighter(self):
        """Very dark image should be brightened after CLAHE."""
        from src.preprocessing.contrast import fix_lighting
        dark = np.full((100, 100, 3), 10, dtype=np.uint8)
        bright = fix_lighting(dark)
        assert bright.mean() > dark.mean(), "CLAHE should increase mean brightness on dark image."


class TestAdaptiveBinarize:
    def test_output_is_binary(self):
        """Output should be a grayscale image with only values 0 and 255."""
        from src.preprocessing.contrast import adaptive_binarize
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = adaptive_binarize(img)
        unique = set(np.unique(result))
        assert unique <= {0, 255}, f"Non-binary values: {unique}"

    def test_grayscale_input_accepted(self):
        """Should accept pre-converted grayscale arrays."""
        from src.preprocessing.contrast import adaptive_binarize
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = adaptive_binarize(gray)
        assert result.shape == (100, 100)
