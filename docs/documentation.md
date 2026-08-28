# Receipt OCR Pipeline — Technical Documentation

## 1. Approach

### Problem & Goals
The task is to extract structured financial data (store name, date, line items, total) from noisy real-world receipt images and produce confidence-scored JSON outputs alongside a cross-receipt financial summary.

### Pipeline Design
We chose a **rule-based extraction pipeline** over an end-to-end deep learning approach for the following reasons:
- **Explainability**: every extracted field has a traceable confidence formula with interpretable components (OCR quality, regex match, positional heuristic).
- **Build speed**: the full system can be constructed and debugged within the time-boxed assignment window without requiring labelled training data.
- **Tunability**: all thresholds and weights are centralised in `config.py`, making it trivial to tune without touching extraction logic.

### Pipeline Stages

```
Raw Image → Preprocessing → OCR Engine → Field Extraction
         → Confidence Scoring → JSON Structuring → Financial Summary
```

**Stage 1 — Preprocessing** (`src/preprocessing/`)
Applied in a fixed order that maximises OCR accuracy:
1. Non-local means denoising (removes JPEG/thermal-print noise without blurring text edges)
2. Laplacian variance blur detection → sharpening if blurry
3. Skew correction via Otsu threshold + `cv2.minAreaRect` angle estimation
4. CLAHE contrast enhancement on the LAB luminance channel (handles shadows and glare without over-saturating bright regions)
5. Gaussian adaptive binarisation (handles uneven illumination across the receipt)

**Stage 2 — OCR** (`src/ocr/`)
- **Primary**: EasyOCR with `gpu=False` (works out-of-box on CPU, returns per-word confidence scores, handles varied fonts and minor rotations better than Tesseract on raw images).
- **Fallback**: Tesseract (`pytesseract.image_to_data`) is invoked if EasyOCR's mean confidence falls below 0.4. Results from both engines are normalised to the same token schema; whichever has higher mean confidence is used.
- OCR output is cached to `outputs/ocr_raw/` as JSON so debugging extraction failures doesn't require expensive re-OCR.
- Tokens are sorted into reading order (top-to-bottom, left-to-right by bounding box coordinates) before any extraction.

**Stage 3 — Field Extraction** (`src/extraction/`)
Four independent, unit-testable extractors:

| Field | Strategy |
|---|---|
| Store name | Top-3 tokens by tallest bounding box (font size proxy for header text) |
| Date | Multi-pattern regex bank → `dateutil` validation → ISO 8601 normalisation |
| Line items | Line grouping by vertical proximity → regex pattern match → non-item keyword filter |
| Total amount | Bottom-up keyword adjacency search → multi-candidate scoring → tie-break by position + magnitude |

**Stage 4 — Confidence Scoring** (`src/confidence/scorer.py`)

```
field_confidence = 0.50 × ocr_confidence
                 + 0.30 × pattern_validity
                 + 0.20 × heuristic_score
```

Fields with confidence below 0.70 are flagged with a human-readable reason string.  Total amount extraction additionally exposes `alternative_candidates` for full transparency when multiple totals are detected.

---

## 2. Tools Used

| Tool | Version | Purpose |
|---|---|---|
| EasyOCR | ≥1.7 | Primary OCR engine |
| pytesseract | ≥0.3 | Fallback OCR engine |
| OpenCV | ≥4.8 | Image preprocessing |
| NumPy | ≥1.24 | Array operations |
| python-dateutil | ≥2.8 | Robust date parsing / validation |
| jsonschema | ≥4.19 | Receipt output validation |
| Pillow | ≥10.0 | Image I/O for Tesseract |
| pandas | ≥2.0 | Data audit spreadsheet loading |
| regex | ≥2023 | Extended regex for price/date patterns |

---

## 3. Challenges & Solutions

### Skew Detection Instability
`cv2.minAreaRect` on sparse-text receipts (e.g. a receipt with only 2–3 lines of text) can return unstable angles, sometimes near ±45°. **Solution**: added a minimum foreground-pixel count guard (< 5 pixels → return original unchanged) and flagged any `|angle| > 30°` as extreme for manual review.

### Ambiguous Total vs. Subtotal
Receipts frequently list Subtotal, Tax, and Total in close proximity. A naive "first number near keyword" approach picks the wrong value. **Solution**: multi-candidate scoring combining keyword specificity (grand total > total > subtotal), position from bottom, and magnitude (larger amounts preferred). Alternative candidates are surfaced in the output for transparency.

### Thermal-Print Image Quality
Faded thermal receipts collapse OCR confidence for all engines. **Solution**: CLAHE preprocessing significantly improves legibility; additionally, if mean confidence falls below 0.3, fine-grained item extraction is skipped (still attempt store/date/total) and the receipt is marked `"quality": "low"` rather than crashing.

### European Price Format
Comma-decimal prices (e.g. `2,99` in German/French receipts) break standard float parsing. **Solution**: `_normalise_price()` detects the European thousands+decimal pattern and converts to period-decimal before any numeric operations.

---

## 4. Future Improvements

1. **Layout-aware models**: Replace rule-based extraction with LayoutLMv3 or Donut, which process the image + OCR tokens jointly and produce key-value pairs without hand-written regex. These models handle arbitrary receipt layouts including two-column and rotated formats far better.

2. **EasyOCR fine-tuning**: Fine-tune EasyOCR's recognition model on a labelled subset of receipt crops (30–50 examples are sufficient for meaningful improvement on domain-specific fonts like thermal-print and dot-matrix).

3. **Active learning loop**: Route low-confidence fields to a lightweight human review interface. Reviewed corrections feed back into fine-tuning data, gradually reducing the flagged proportion over time.

4. **Quantitative evaluation**: Build a small manually-labelled validation set (50–100 receipts) and track field-level F1 and character error rate before/after each preprocessing improvement.

5. **Multi-language support**: Extend EasyOCR language list to cover Arabic, Hindi (Devanagari), and Chinese receipts, which the current pipeline marks as "unsupported".
