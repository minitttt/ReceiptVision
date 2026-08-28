"""
summary/aggregator.py — Financial summary generation across all receipts.
"""

import json
import logging
from pathlib import Path
import re
from typing import Optional

import config

logger = logging.getLogger(__name__)


def _normalise_store_name(name: str) -> str:
    """
    Clean up and standardise store names for aggregation.
    Groups common variants (e.g., WAL*MART, Walmart, WALAMART).
    """
    if not name:
        return "Unknown Store"

    clean = re.sub(r'[^a-zA-Z0-9\s]', '', name).strip().upper()

    if "WAL" in clean and "MART" in clean:
        return "Walmart"
    if "DOLLAR" in clean and ("TREE" in clean or "TNEE" in clean):
        return "Dollar Tree"
    if "TRADER" in clean and "JOE" in clean:
        return "Trader Joe's"
    if "WHOLE FOODS" in clean:
        return "Whole Foods"
    if "TARGET" in clean:
        return "Target"

    return name.title()


def generate_summary(
    all_receipts: list[dict],
    output_dir: Optional[str] = None,
) -> dict:
    """
    Aggregate financial statistics across all processed receipts.

    Receipts with a null or unparseable total_amount are excluded from
    monetary aggregations and reported separately in ``unprocessed_receipts``
    so totals are never silently wrong.

    Args:
        all_receipts: List of receipt dicts (from ``build_receipt_json``).
        output_dir: Where to write ``financial_summary.json``.
                    Defaults to ``config.OUTPUT_SUMMARY_DIR``.

    Returns:
        Financial summary dict.
    """
    output_dir = output_dir or config.OUTPUT_SUMMARY_DIR

    total_spend = 0.0
    spend_per_store: dict[str, float] = {}
    processed = []
    unprocessed = []
    low_quality = []

    for receipt in all_receipts:
        receipt_id = receipt.get("receipt_id", "unknown")
        quality = receipt.get("quality", "ok")

        if quality == "low":
            low_quality.append(receipt_id)

        total_value = receipt.get("total_amount")

        if total_value is None:
            flags = receipt.get("metadata", {}).get("flags", {})
            reason = flags.get("total_amount", "total_amount is null")
            unprocessed.append({
                "receipt_id": receipt_id,
                "reason": reason,
            })
            continue

        try:
            amount = float(total_value)
        except (ValueError, TypeError):
            unprocessed.append({
                "receipt_id": receipt_id,
                "reason": f"unparseable total_amount: {total_value!r}",
            })
            continue

        store = _normalise_store_name(receipt.get("store_name"))

        total_spend += amount
        spend_per_store[store] = round(spend_per_store.get(store, 0.0) + amount, 2)
        processed.append(receipt_id)

    summary = {
        "total_spend": round(total_spend, 2),
        "number_of_transactions": len(all_receipts),
        "processed_receipts": len(processed),
        "unprocessed_receipts": unprocessed,
        "low_quality_receipts": low_quality,
        "spend_per_store": spend_per_store,
        "average_transaction_value": (
            round(total_spend / len(processed), 2) if processed else None
        ),
    }

    _write_summary(summary, output_dir)
    return summary


def _write_summary(summary: dict, output_dir: str) -> None:
    """Write the financial summary to disk."""
    out_path = Path(output_dir) / "financial_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Financial summary written: %s", out_path)


def load_all_receipts(json_dir: Optional[str] = None) -> list[dict]:
    """
    Load all per-receipt JSON files from ``json_dir``.

    Args:
        json_dir: Directory containing ``*.json`` receipt files.
                  Defaults to ``config.OUTPUT_JSON_DIR``.

    Returns:
        List of receipt dicts.
    """
    json_dir = json_dir or config.OUTPUT_JSON_DIR
    receipts = []
    for path in sorted(Path(json_dir).glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                receipts.append(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load %s: %s", path, exc)
    logger.info("Loaded %d receipt files from %s", len(receipts), json_dir)
    return receipts
