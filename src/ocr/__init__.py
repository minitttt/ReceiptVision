"""src/ocr package."""
from .engine import run_easyocr
from .fallback_tesseract import run_tesseract

__all__ = ["run_easyocr", "run_tesseract"]
