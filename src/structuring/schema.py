"""
structuring/schema.py — Receipt JSON schema assembly and validation.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import jsonschema

import config

logger = logging.getLogger(__name__)


def build_receipt_json(
    receipt_id: str,
    quality: str,
    store_field: dict,
    date_field: dict,
    items_fields: list[dict],
    total_field: dict,
    preprocessing_meta: Optional[dict] = None,
) -> dict:
    """
    Assemble the complete per-receipt JSON object.

    Args:
        receipt_id: Unique identifier (typically the image filename stem).
        quality: ``"ok"``, ``"low"``, or ``"unsupported"``.
        store_field: Confidence-scored store_name field dict.
        date_field: Confidence-scored date field dict.
        items_fields: List of confidence-scored item dicts.
        total_field: Confidence-scored total_amount field dict.
        preprocessing_meta: Optional metadata from preprocessing
            (blur_score, skew_angle, etc.).

    Returns:
        Receipt dict conforming to ``config.RECEIPT_SCHEMA``.
    """
    avg_item_conf = 0.0
    if items_fields:
        avg_item_conf = sum(i.get("confidence", 0.0) for i in items_fields) / len(items_fields)

    receipt = {
        "receipt_id": receipt_id,
        "quality": quality,
        "store_name": store_field.get("value"),
        "date": date_field.get("value"),
        "items": [{"name": i.get("name", ""), "price": i.get("price")} for i in items_fields],
        "total_amount": total_field.get("value"),
        "confidence_scores": {
            "store_name": store_field.get("confidence", 0.0),
            "date": date_field.get("confidence", 0.0),
            "items": round(avg_item_conf, 3),
            "total_amount": total_field.get("confidence", 0.0),
        },
    }

    metadata = {}
    if preprocessing_meta:
        metadata["preprocessing"] = preprocessing_meta

    flags = {}
    for k, field in [("store_name", store_field), ("date", date_field), ("total_amount", total_field)]:
        if field.get("flagged"):
            flags[k] = field.get("reason")
        if k == "total_amount" and field.get("alternative_candidates"):
            metadata["alternative_candidates"] = field.get("alternative_candidates")
            
    for i, item in enumerate(items_fields):
        if item.get("flagged"):
            flags[f"item_{i}"] = item.get("reason")
            
    if flags:
        metadata["flags"] = flags

    if metadata:
        receipt["metadata"] = metadata

    return receipt


def validate_schema(receipt: dict) -> tuple[bool, Optional[str]]:
    """
    Validate a receipt dict against the canonical JSON schema.

    Args:
        receipt: Receipt dict to validate.

    Returns:
        Tuple of (is_valid, error_message). ``error_message`` is None on success.
    """
    try:
        jsonschema.validate(instance=receipt, schema=config.RECEIPT_SCHEMA)
        return True, None
    except jsonschema.ValidationError as exc:
        return False, exc.message


def write_receipt_json(receipt: dict, output_dir: str = None) -> Path:
    """
    Validate and write a receipt dict to ``<output_dir>/<receipt_id>.json``.

    Args:
        receipt: Receipt dict (from ``build_receipt_json``).
        output_dir: Target directory. Defaults to ``config.OUTPUT_JSON_DIR``.

    Returns:
        Path to the written file.

    Raises:
        ValueError: If schema validation fails.
    """
    output_dir = output_dir or config.OUTPUT_JSON_DIR
    is_valid, err = validate_schema(receipt)
    if not is_valid:
        logger.error("Schema validation failed for %s: %s", receipt.get("receipt_id"), err)
        raise ValueError(f"Schema validation failed: {err}")

    out_path = Path(output_dir) / f"{receipt['receipt_id']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)

    logger.info("Receipt JSON written: %s", out_path)
    return out_path
