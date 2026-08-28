"""
config.py — Central configuration for the Receipt OCR pipeline.

All thresholds, regex patterns, and confidence weights live here.
Change values here to tune the pipeline without touching source code.
"""

OCR_LANGUAGE = ["en"]
OCR_GPU = False

OCR_FALLBACK_THRESHOLD = 0.40

OCR_LOW_QUALITY_THRESHOLD = 0.30

OCR_CACHE_DIR = "outputs/ocr_raw"

BLUR_VARIANCE_THRESHOLD = 100.0   # Laplacian variance; below → sharpen
SKEW_EXTREME_ANGLE = 30.0         # degrees; beyond → flag as extreme skew
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
ADAPTIVE_THRESH_BLOCK_SIZE = 31
ADAPTIVE_THRESH_C = 15

CONF_WEIGHT_OCR = 0.50
CONF_WEIGHT_PATTERN = 0.30
CONF_WEIGHT_HEURISTIC = 0.20

CONF_FLAG_THRESHOLD = 0.70

DATE_PATTERNS = [
    r"\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}",           # [0] 2024-05-12  (ISO 8601 — yearfirst)
    r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}",         # [1] 12/05/2024, 12-05-24
    r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}",           # [2] May 12, 2024
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",              # [3] 12 May 2024
    r"\d{1,2}[A-Za-z]{3}\d{2,4}",                    # [4] 12May2024 (compact)
]

ITEM_PRICE_PATTERN = r"^(.+?)\s+[\$₹€£]?\s?((?:\d{1,3}(?:,\s?\d{3})+(?:\.\d+)?|\d+(?:[.,]\d{2})))\s*$"

NON_ITEM_KEYWORDS = [
    "subtotal", "sub total", "sub-total", "subtota]", "subtota1", "subtota",
    "tax", "vat", "gst", "hst", "pst",
    "total", "grand total", "amount due", "balance due", "tota]", "tota1", "totai", "amount",
    "change", "cash", "credit", "debit", "card",
    "discount", "coupon", "savings", "tip", "gratuity",
    "visa", "mastercard", "amex", "payment",
    "items", "count", "sold", "return"
]

MULTILINE_MERGE_THRESHOLD = 30

TOTAL_KEYWORDS = [
    "grand total", "total amount", "amount due",
    "balance due", "total due", "net total",
    "total", "tota1", "tota]", "totai", "tota!", "tota|", "tota", "amount",
]

PRICE_PATTERN = r"[\$₹€£]?\s?((?:\d{1,3}(?:,\s?\d{3})+(?:\.\d+)?|\d+(?:[.,]\d{2})))"

STORE_TOP_LINES = 3

OUTPUT_JSON_DIR = "outputs/json"
OUTPUT_SUMMARY_DIR = "outputs/summary"
OUTPUT_LOGS_DIR = "outputs/logs"
DATA_PROCESSED_DIR = "data/processed"

RECEIPT_SCHEMA = {
    "type": "object",
    "required": ["receipt_id", "quality", "store_name", "date", "items", "total_amount"],
    "properties": {
        "receipt_id": {"type": "string"},
        "quality": {"type": "string", "enum": ["ok", "low", "unsupported"]},
        "store_name": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"]},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "price"],
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": ["string", "null"]},
                },
            },
        },
        "total_amount": {"type": ["string", "null"]},
        "confidence_scores": {"type": "object"},
    },
}
