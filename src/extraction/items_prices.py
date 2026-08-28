"""
extraction/items_prices.py — Line-item and price extraction.
"""

import re
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)

_ITEM_PATTERN = re.compile(config.ITEM_PRICE_PATTERN, re.IGNORECASE)
_NON_ITEM_RE = re.compile(
    r"(?<![a-zA-Z])(" + "|".join(re.escape(kw) for kw in config.NON_ITEM_KEYWORDS) + r")(?![a-zA-Z])",
    re.IGNORECASE,
)
_PRICE_ONLY_RE = re.compile(
    r"^[\$₹€£]?\s?((?:\d{1,3}(?:,\s?\d{3})+(?:\.\d+)?|\d+(?:[.,]\d{2})))\s*$"
)


def _is_non_item(text: str) -> bool:
    """Return True if the line is a subtotal / tax / total line."""
    return bool(_NON_ITEM_RE.search(text))


def _normalise_price(raw: str) -> str:
    """Convert OCR money text to a consistent period-decimal string."""
    raw = raw.replace(" ", "")
    if re.match(r"^\d{1,3}(,\d{3})+(\.\d+)?$", raw):
        return f"{float(raw.replace(',', '')):.2f}"
    if re.match(r"^\d{1,3}(\.\d{3})*,\d{2}$", raw):
        return f"{float(raw.replace('.', '').replace(',', '.')):.2f}"
    return f"{float(raw.replace(',', '.')):.2f}"


def _group_into_lines(tokens: list[dict], vertical_gap: float = None) -> list[list[dict]]:
    """
    Group OCR tokens into logical lines based on vertical proximity.

    Tokens whose top-y coordinates are within ``vertical_gap`` pixels of
    each other are considered to be on the same line.

    Args:
        tokens: Sorted OCR tokens.
        vertical_gap: Max pixel gap between tokens on the same line.

    Returns:
        List of groups, each group being a list of tokens on one line.
    """
    vertical_gap = vertical_gap or 15.0
    if not tokens:
        return []

    lines = []
    current_line = [tokens[0]]
    current_y = tokens[0]["bbox"][0][1]

    for token in tokens[1:]:
        token_y = token["bbox"][0][1]
        if abs(token_y - current_y) <= vertical_gap:
            current_line.append(token)
        else:
            lines.append(current_line)
            current_line = [token]
            current_y = token_y

    lines.append(current_line)
    return lines


def _line_text_and_conf(line_tokens: list[dict]) -> tuple[str, float]:
    """Join tokens into a line string and compute average confidence."""
    text = " ".join(t["text"] for t in line_tokens).strip()
    conf = sum(t["confidence"] for t in line_tokens) / len(line_tokens)
    return text, conf


def extract_items(tokens: list[dict]) -> list[dict]:
    """
    Extract line items (name + price) from OCR tokens.

    Strategy:
    1. Group tokens into visual lines.
    2. Filter out header/footer lines (non-item keywords).
    3. Apply ``ITEM_PRICE_PATTERN`` to each line.
    4. Handle multi-line items where the name and price appear on adjacent
       lines (detected by the price-only pattern on the following line).

    Args:
        tokens: Sorted OCR token dicts.

    Returns:
        List of item dicts: ``{name, price, confidence, raw_line}``.
    """
    lines = _group_into_lines(tokens)
    items = []
    skip_next = False

    for i, line_tokens in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        text, conf = _line_text_and_conf(line_tokens)

        if _is_non_item(text):
            continue

        match = _ITEM_PATTERN.match(text)
        if match:
            name = match.group(1).strip()
            price = _normalise_price(match.group(2))
            items.append({
                "name": name,
                "price": price,
                "confidence": round(conf, 4),
                "raw_line": text,
            })
            continue

        if i + 1 < len(lines):
            next_text, next_conf = _line_text_and_conf(lines[i + 1])
            price_match = _PRICE_ONLY_RE.match(next_text)
            if price_match and not _is_non_item(next_text):
                curr_bottom_y = max(t["bbox"][2][1] for t in line_tokens)
                next_top_y = min(t["bbox"][0][1] for t in lines[i + 1])
                gap = next_top_y - curr_bottom_y
                if gap <= config.MULTILINE_MERGE_THRESHOLD:
                    price = _normalise_price(price_match.group(1))
                    avg_conf = (conf + next_conf) / 2
                    items.append({
                        "name": text.strip(),
                        "price": price,
                        "confidence": round(avg_conf, 4),
                        "raw_line": f"{text} | {next_text}",
                    })
                    skip_next = True
                    continue

        logger.debug("Skipping unmatched line: '%s'", text)

    logger.info("Extracted %d line items.", len(items))
    return items
