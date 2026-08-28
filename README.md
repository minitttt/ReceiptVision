# ReceiptVision
## Receipt OCR and Expense Analytics

An end-to-end pipeline for turning receipt images into structured JSON and financial summaries.

---

## Architecture

```
Raw Image
   │
   ▼
[1] Preprocessing   denoise → deskew → CLAHE contrast → adaptive binarize
   │
   ▼
[2] OCR Engine      EasyOCR (primary) + Tesseract (fallback if conf < 0.4)
   │
   ▼
[3] Field Extraction store_name · date · items+prices · total_amount
   │
   ▼
[4] Confidence Scoring  0.5×ocr + 0.3×pattern_validity + 0.2×heuristic
   │
   ▼
[5] JSON Structuring    jsonschema-validated per-receipt output
   │
   ▼
[6] Financial Summary   spend_per_store · total_spend · unprocessed list
```

---

## Setup

### 1. Prerequisites
```bash
# macOS
brew install tesseract          # required for fallback OCR engine

# Linux
sudo apt-get install tesseract-ocr
```

### 2. Python environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Dataset
Place receipt images in `data/raw/`. Supported formats are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, and `.webp`.

The current dataset contains sequentially named files from `1.jpg` to `371.jpg`.

---

## Usage

### Process all receipts in `data/raw`
```bash
python src/pipeline.py --input data/raw --output outputs
```

This also saves original, processed, and binarized images in `data/processed/`.

### Process the first 100 receipts
```bash
python src/pipeline.py --input data/sample100 --output outputs
```

### Process a single receipt
```bash
python src/pipeline.py --input data/raw/1.jpg
```

### Regenerate summary from existing JSON outputs (without re-running OCR)
```bash
python src/pipeline.py --input data/raw --output outputs --summary-only
```


## Output Structure

```
outputs/
├── json/
│   ├── receipt_001.json        # per-receipt structured data
│   └── receipt_002.json
├── summary/
│   └── financial_summary.json  # aggregated financial stats
├── ocr_raw/
│   └── receipt_001_easyocr.json  # raw OCR cache (for debugging)
└── logs/
    ├── pipeline_run.log
    └── receipt_001.log
```

### Per-receipt JSON schema
```json
{
  "receipt_id": "receipt_001",
  "quality": "ok",
  "store_name": "Walmart",
  "date": "2024-05-12",
  "items": [
    {
      "name": "Milk 2%",
      "price": "3.49"
    }
  ],
  "total_amount": "24.99",
  "confidence_scores": {
    "store_name": 0.93,
    "date": 0.88,
    "items": 0.81,
    "total_amount": 0.96
  },
  "metadata": {
    "preprocessing": {
      "blur_score": 142.7,
      "skew_angle": -1.3,
      "extreme_skew": false,
      "ocr_engine": "easyocr",
      "mean_ocr_confidence": 0.87
    },
    "flags": {},
    "alternative_candidates": []
  }
}
```

### Financial summary schema
```json
{
  "total_spend": 156.23,
  "number_of_transactions": 8,
  "processed_receipts": 7,
  "unprocessed_receipts": [{"receipt_id": "...", "reason": "..."}],
  "low_quality_receipts": ["receipt_005"],
  "spend_per_store": {"Walmart": 89.12, "Trader Joe's": 67.11},
  "average_transaction_value": 22.32
}
```

---

## Running Tests

```bash
# All tests (no EasyOCR, Tesseract, or real images needed)
python -m pytest tests/ -v

# Individual modules
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_extraction.py -v
python -m pytest tests/test_confidence.py -v
```

## Evaluation Results

The latest run used the first 100 receipts in `data/sample100`.

| Category | Weight | Current result |
|---|---:|---|
| Extraction Accuracy | 30% | Not objectively measured; ground truth is required. Total-field coverage was 60/100. |
| Robustness to Real-World Noise | 15% | Preprocessing includes denoising, blur detection, deskewing, CLAHE, and adaptive binarization. |
| Data Structuring | 10% | 100/100 outputs passed JSON schema validation. |
| Financial Summary Logic | 10% | Summary reconciles with per-receipt totals and excludes null totals. |
| Confidence Scoring | 20% | Field-level weighted scoring and review flags are included in every output. |
| Code Quality | 10% | Modular extractors, centralized configuration, CLI, logging, and tests. |
| Edge Case Handling | 5% | Handles missing OCR fields, unsupported images, low confidence, malformed amounts, and barcode false positives. |

### Current run metrics

```text
Receipts processed:       100
Store names extracted:    100/100
Dates extracted:           82/100
Totals extracted:          60/100
Receipts with items:       61/100
Schema-valid JSON files:  100/100
Null totals:               40/100
Total spend:               178722.20
```

These are coverage and validation metrics, not field-level accuracy. To calculate actual accuracy, add manually verified values for a representative set of receipts and compare them with the generated JSON.

---

## Configuration

All tunable parameters live in [`config.py`](config.py):

| Parameter | Default | Description |
|---|---|---|
| `OCR_FALLBACK_THRESHOLD` | `0.40` | EasyOCR mean conf below which Tesseract is tried |
| `OCR_LOW_QUALITY_THRESHOLD` | `0.30` | Below this → receipt marked "low" quality |
| `BLUR_VARIANCE_THRESHOLD` | `100.0` | Laplacian variance below which sharpening is applied |
| `CONF_FLAG_THRESHOLD` | `0.70` | Field confidence below this → flagged |
| `CONF_WEIGHT_OCR` | `0.50` | Weight of OCR confidence in composite score |
| `CONF_WEIGHT_PATTERN` | `0.30` | Weight of pattern validity |
| `CONF_WEIGHT_HEURISTIC` | `0.20` | Weight of positional heuristic |

---

## Repository Structure

```
carbon-crunch-ocr/
├── config.py                       # all thresholds, patterns, weights
├── requirements.txt
├── src/
│   ├── pipeline.py                 # single orchestrator + CLI
│   ├── preprocessing/
│   │   ├── denoise.py              # NL-means + blur metric
│   │   ├── deskew.py               # skew detection + correction
│   │   └── contrast.py             # CLAHE + adaptive binarize
│   ├── ocr/
│   │   ├── engine.py               # EasyOCR wrapper + cache
│   │   └── fallback_tesseract.py   # Tesseract normalised wrapper
│   ├── extraction/
│   │   ├── store_name.py
│   │   ├── date_extractor.py
│   │   ├── items_prices.py
│   │   └── total_amount.py
│   ├── confidence/
│   │   └── scorer.py               # weighted formula + flagging
│   ├── structuring/
│   │   └── schema.py               # JSON assembly + validation
│   └── summary/
│       └── aggregator.py           # financial aggregation
├── tests/
│   ├── test_preprocessing.py
│   ├── test_extraction.py
│   └── test_confidence.py
├── data/
│   ├── raw/                        # ← put receipt images here
│   └── processed/                  # debug preprocessing images
├── outputs/
│   ├── json/
│   ├── summary/
│   ├── ocr_raw/
│   └── logs/
├── notebooks/
│   └── exploration.ipynb
└── docs/
    └── documentation.md
```

---

## Edge Case Handling

| Scenario | Behaviour |
|---|---|
| Image fails to load | Empty receipt JSON with `"quality": "unsupported"` |
| EasyOCR conf < 0.4 | Tesseract fallback attempted; best engine used |
| OCR conf < 0.3 | Marked `"quality": "low"`; item extraction skipped |
| Field extraction fails | Field defaults to `{"value": null, "flagged": true, "reason": "..."}` |
| Multiple total candidates | Best scored returned; alternatives in `alternative_candidates` |
| `|skew| > 30°` | Logged as extreme; still processed |
| Unsupported / handwritten | OCR confidence collapse detected; marked unsupported |
