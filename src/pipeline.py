"""
pipeline.py — End-to-end receipt OCR processing pipeline orchestrator.

Usage (CLI):
    python src/pipeline.py --input data/raw/ --output outputs/

Usage (Python API):
    from src.pipeline import process_receipt
    result = process_receipt("data/raw/my_receipt.jpg")
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config
from src.preprocessing.contrast import preprocess_pipeline
from src.ocr.engine import run_easyocr
from src.ocr.fallback_tesseract import run_tesseract
from src.extraction.store_name import extract_store_name
from src.extraction.date_extractor import extract_date
from src.extraction.items_prices import extract_items
from src.extraction.total_amount import extract_total
from src.confidence.scorer import build_field, score_items
from src.structuring.schema import build_receipt_json, write_receipt_json
from src.summary.aggregator import generate_summary, load_all_receipts

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _setup_logging(log_dir: str, receipt_id: str = "pipeline") -> None:
    """Configure logging to both console and per-run log file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{receipt_id}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if not root.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(fh)


logger = logging.getLogger(__name__)


def process_receipt(
    image_path: str,
    output_dir: str = "outputs",
    save_debug: bool = True,
) -> dict:
    """
    Run the full OCR pipeline for a single receipt image.

    This function is the canonical single entry-point for the entire system.
    It handles all exceptions internally — any receipt always produces a JSON
    output, even if it's mostly null/low-confidence.

    Args:
        image_path: Path to the receipt image file.
        output_dir: Base output directory (sub-dirs are created automatically).
        save_debug: Whether to save preprocessed debug images.

    Returns:
        Receipt dict conforming to the output schema.
    """
    image_path = Path(image_path)
    receipt_id = image_path.stem

    json_dir = str(Path(output_dir) / "json")
    log_dir = str(Path(output_dir) / "logs")
    summary_dir = str(Path(output_dir) / "summary")
    ocr_cache_dir = str(Path(output_dir) / "ocr_raw")
    processed_dir = "data/processed"

    _setup_logging(log_dir, receipt_id)
    logger.info("=" * 60)
    logger.info("Processing receipt: %s", image_path)
    logger.info("=" * 60)

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"cv2.imread returned None for: {image_path}")
    except Exception as exc:
        logger.error("Failed to load image %s: %s", image_path, exc)
        return _make_empty_receipt(receipt_id, reason=f"image load error: {exc}")

    try:
        prep_result = preprocess_pipeline(
            img,
            receipt_id=receipt_id,
            processed_dir=processed_dir,
            blur_threshold=config.BLUR_VARIANCE_THRESHOLD,
            clahe_clip=config.CLAHE_CLIP_LIMIT,
            clahe_tile=config.CLAHE_TILE_GRID,
            adaptive_block=config.ADAPTIVE_THRESH_BLOCK_SIZE,
            adaptive_c=config.ADAPTIVE_THRESH_C,
            skew_extreme=config.SKEW_EXTREME_ANGLE,
            save_debug=save_debug,
        )
        processed_img = prep_result["processed"]
        prep_meta = prep_result["metadata"]
        blur_score = prep_result["blur_score"]
        logger.info("[%s] Preprocessing done. blur=%.2f skew=%.2f° extreme=%s",
                    receipt_id, blur_score, prep_result["skew_angle"], prep_result["extreme_skew"])
    except Exception as exc:
        logger.error("[%s] Preprocessing failed: %s\n%s", receipt_id, exc, traceback.format_exc())
        processed_img = img
        prep_meta = {}
        blur_score = 0.0

    try:
        tokens, mean_conf = run_easyocr(processed_img, receipt_id, cache_dir=ocr_cache_dir)
        ocr_engine = "easyocr"

        if mean_conf < config.OCR_FALLBACK_THRESHOLD:
            logger.info("[%s] EasyOCR mean_conf=%.3f below threshold %.2f — trying Tesseract fallback.",
                        receipt_id, mean_conf, config.OCR_FALLBACK_THRESHOLD)
            tess_tokens, tess_conf = run_tesseract(processed_img, receipt_id)
            if tess_conf > mean_conf:
                tokens, mean_conf = tess_tokens, tess_conf
                ocr_engine = "tesseract"
                logger.info("[%s] Using Tesseract results (conf=%.3f > easyocr conf).", receipt_id, mean_conf)

        logger.info("[%s] OCR engine='%s', tokens=%d, mean_conf=%.3f",
                    receipt_id, ocr_engine, len(tokens), mean_conf)
    except Exception as exc:
        logger.error("[%s] OCR failed: %s\n%s", receipt_id, exc, traceback.format_exc())
        return _make_empty_receipt(receipt_id, reason=f"OCR error: {exc}", prep_meta=prep_meta)

    if mean_conf < config.OCR_LOW_QUALITY_THRESHOLD:
        logger.warning("[%s] Very low OCR confidence (%.3f). Marking as 'low' quality.", receipt_id, mean_conf)
        quality = "low"
    elif not tokens:
        logger.warning("[%s] No OCR tokens found. Marking as 'unsupported'.", receipt_id)
        return _make_empty_receipt(receipt_id, reason="no OCR tokens extracted", prep_meta=prep_meta)
    else:
        quality = "ok"

    store_value, store_evidence = _safe_extract(extract_store_name, tokens, receipt_id, "store_name")
    date_value, date_evidence = _safe_extract(extract_date, tokens, receipt_id, "date")
    total_value, total_evidence = _safe_extract(extract_total, tokens, receipt_id, "total_amount")

    if quality == "low":
        raw_items = []
        logger.info("[%s] Skipping item extraction (low quality).", receipt_id)
    else:
        raw_items = _safe_extract_items(tokens, receipt_id)

    store_field = build_field(
        value=store_value,
        ocr_confidence=store_evidence.get("ocr_confidence", 0.0),
        pattern_validity=1.0 if store_value else 0.0,
        heuristic_score=store_evidence.get("heuristic_score", 0.5),
    )

    date_field = build_field(
        value=date_value,
        ocr_confidence=date_evidence.get("ocr_confidence", 0.0),
        pattern_validity=date_evidence.get("pattern_validity", 0.0),
        heuristic_score=0.8 if date_value else 0.0,
    )

    total_field = build_field(
        value=total_value,
        ocr_confidence=total_evidence.get("ocr_confidence", 0.0),
        pattern_validity=1.0 if total_value else 0.0,
        heuristic_score=total_evidence.get("heuristic_score", 0.0),
        extra={
            "alternative_candidates": total_evidence.get("alternative_candidates", []),
        } if total_evidence.get("alternative_candidates") else None,
    )

    scored_items = score_items(raw_items)

    receipt = build_receipt_json(
        receipt_id=receipt_id,
        quality=quality,
        store_field=store_field,
        date_field=date_field,
        items_fields=scored_items,
        total_field=total_field,
        preprocessing_meta={
            **prep_meta,
            "ocr_engine": ocr_engine,
            "mean_ocr_confidence": round(mean_conf, 4),
            "token_count": len(tokens),
        },
    )

    try:
        write_receipt_json(receipt, output_dir=json_dir)
    except ValueError as exc:
        logger.error("[%s] Schema validation error: %s", receipt_id, exc)

    logger.info("[%s] Done. quality=%s store='%s' date=%s total=%s items=%d",
                receipt_id, quality,
                store_field.get("value"), date_field.get("value"),
                total_field.get("value"), len(scored_items))
    return receipt


def _safe_extract(fn, tokens, receipt_id, field_name):
    """Call an extractor; return (None, {}) on any exception."""
    try:
        return fn(tokens)
    except Exception as exc:
        logger.error("[%s] %s extraction error: %s\n%s", receipt_id, field_name, exc, traceback.format_exc())
        return None, {}


def _safe_extract_items(tokens, receipt_id):
    """Call extract_items; return [] on any exception."""
    try:
        from src.extraction.items_prices import extract_items
        return extract_items(tokens)
    except Exception as exc:
        logger.error("[%s] items extraction error: %s\n%s", receipt_id, exc, traceback.format_exc())
        return []


def _make_empty_receipt(receipt_id: str, reason: str = "unknown error", prep_meta: dict = None) -> dict:
    """
    Produce a valid but empty receipt dict with all fields null and flagged.

    Guarantees the pipeline always returns a JSON-serialisable dict.
    """
    return {
        "receipt_id": receipt_id,
        "quality": "unsupported",
        "store_name": None,
        "date": None,
        "items": [],
        "total_amount": None,
        "confidence_scores": {
            "store_name": 0.0,
            "date": 0.0,
            "items": 0.0,
            "total_amount": 0.0,
        },
        "metadata": {
            "preprocessing": prep_meta or {},
            "flags": {
                "store_name": reason,
                "date": reason,
                "total_amount": reason,
            },
        },
        "error": reason,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Carbon Crunch — Receipt OCR Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py --input data/raw/

  python src/pipeline.py --input data/raw/ --output my_outputs/

  python src/pipeline.py --input data/raw/receipt_01.jpg

  python src/pipeline.py --input data/raw/ --no-debug
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to a single image file or a directory of images.",
    )
    parser.add_argument(
        "--output", "-o", default="outputs",
        help="Base output directory (default: outputs/).",
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Skip saving preprocessed debug images.",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip OCR — regenerate financial summary from existing JSON outputs.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    save_debug = not args.no_debug

    _setup_logging(str(Path(args.output) / "logs"), "pipeline_run")
    logger.info("Starting Carbon Crunch Receipt OCR Pipeline")
    logger.info("Input: %s | Output: %s", input_path, args.output)

    if args.summary_only:
        logger.info("Summary-only mode: loading existing receipts from %s/json/", args.output)
        receipts = load_all_receipts(json_dir=str(Path(args.output) / "json"))
        summary = generate_summary(receipts, output_dir=str(Path(args.output) / "summary"))
        print(json.dumps(summary, indent=2))
        return

    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    if not image_files:
        logger.warning("No image files found in %s", input_path)
        sys.exit(0)

    logger.info("Found %d image(s) to process.", len(image_files))

    all_receipts = []
    for img_path in image_files:
        receipt = process_receipt(str(img_path), output_dir=args.output, save_debug=save_debug)
        all_receipts.append(receipt)

    summary = generate_summary(all_receipts, output_dir=str(Path(args.output) / "summary"))

    print("\n" + "=" * 50)
    print("FINANCIAL SUMMARY")
    print("=" * 50)
    print(json.dumps(summary, indent=2))
    print(f"\nPer-receipt JSONs → {args.output}/json/")
    print(f"Summary          → {args.output}/summary/financial_summary.json")
    print(f"Logs             → {args.output}/logs/")


if __name__ == "__main__":
    main()
