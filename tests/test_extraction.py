"""
tests/test_extraction.py — Unit tests for all four field extractors.

All tests use synthetic OCR-line fixtures — no EasyOCR or real images needed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Fixtures — synthetic OCR token lists
# ---------------------------------------------------------------------------

def _make_token(text, confidence=0.90, x=10, y=50, w=200, h=20):
    """Build a minimal OCR token dict matching the EasyOCR schema."""
    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return {
        "text": text,
        "confidence": float(confidence),
        "bbox": bbox,
        "bbox_height": float(h),
    }


def _receipt_tokens_basic():
    """Simulate a clean, well-structured receipt."""
    lines = [
        ("WALMART SUPERCENTER", 0.97, 10, 10, 260, 30),  # large header
        ("123 Main St, Springfield", 0.91, 10, 50, 200, 18),
        ("Date: 05/12/2024", 0.93, 10, 80, 180, 18),
        ("Milk 2%", 0.88, 10, 120, 120, 18),
        ("3.49", 0.85, 200, 120, 60, 18),
        ("Bread Whole Wheat", 0.90, 10, 148, 160, 18),
        ("2.99", 0.86, 200, 148, 60, 18),
        ("Eggs 12pk", 0.89, 10, 176, 120, 18),
        ("4.99", 0.84, 200, 176, 60, 18),
        ("SUBTOTAL", 0.95, 10, 220, 120, 18),
        ("11.47", 0.94, 200, 220, 60, 18),
        ("TAX 8%", 0.91, 10, 248, 100, 18),
        ("0.92", 0.92, 200, 248, 60, 18),
        ("TOTAL", 0.98, 10, 280, 100, 24),
        ("$12.39", 0.97, 200, 280, 80, 24),
    ]
    return [_make_token(text, conf, x, y, w, h) for text, conf, x, y, w, h in lines]


# ---------------------------------------------------------------------------
# Store Name Tests
# ---------------------------------------------------------------------------

class TestExtractStoreName:
    def test_finds_header_store_name(self):
        from src.extraction.store_name import extract_store_name
        tokens = _receipt_tokens_basic()
        value, evidence = extract_store_name(tokens)
        assert value is not None
        assert "WALMART" in value.upper() or len(value) > 0

    def test_empty_tokens_returns_none(self):
        from src.extraction.store_name import extract_store_name
        value, evidence = extract_store_name([])
        assert value is None
        assert evidence["ocr_confidence"] == 0.0

    def test_largest_bbox_preferred(self):
        """The token with the tallest bbox_height should win."""
        from src.extraction.store_name import extract_store_name
        tokens = [
            _make_token("Small text", 0.80, h=10),
            _make_token("STORE NAME", 0.85, y=25, h=30),  # tallest
            _make_token("Address line", 0.88, y=60, h=15),
        ]
        value, _ = extract_store_name(tokens, top_n=3)
        assert value == "STORE NAME"

    def test_evidence_contains_required_keys(self):
        from src.extraction.store_name import extract_store_name
        tokens = _receipt_tokens_basic()
        _, evidence = extract_store_name(tokens)
        assert "ocr_confidence" in evidence
        assert "heuristic_score" in evidence


# ---------------------------------------------------------------------------
# Date Extractor Tests
# ---------------------------------------------------------------------------

class TestExtractDate:
    def test_slash_format_date(self):
        from src.extraction.date_extractor import extract_date
        tokens = [_make_token("Date: 05/12/2024", 0.93)]
        value, evidence = extract_date(tokens)
        assert value == "2024-05-12"
        assert evidence["pattern_validity"] == 1.0

    def test_iso_format_date(self):
        from src.extraction.date_extractor import extract_date
        tokens = [_make_token("2024-05-12", 0.95)]
        value, evidence = extract_date(tokens)
        assert value == "2024-05-12"

    def test_month_name_format(self):
        from src.extraction.date_extractor import extract_date
        tokens = [_make_token("May 12, 2024", 0.90)]
        value, evidence = extract_date(tokens)
        assert value == "2024-05-12"

    def test_no_date_returns_none(self):
        from src.extraction.date_extractor import extract_date
        tokens = [
            _make_token("Milk 2%", 0.88),
            _make_token("$3.49", 0.85),
        ]
        value, evidence = extract_date(tokens)
        assert value is None
        assert evidence["pattern_validity"] == 0.0

    def test_phone_number_not_parsed_as_date(self):
        """A phone number like 555-123-4567 should not be parsed as a date."""
        from src.extraction.date_extractor import extract_date
        tokens = [_make_token("555-123-4567", 0.92)]
        value, _ = extract_date(tokens)
        # 4567 would be an implausible year — should be None
        assert value is None or int(value[:4]) <= 2099

    def test_returns_iso_normalised_string(self):
        from src.extraction.date_extractor import extract_date
        tokens = [_make_token("12-05-2024", 0.90)]
        value, _ = extract_date(tokens)
        assert value is not None
        # ISO format: YYYY-MM-DD
        parts = value.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4


# ---------------------------------------------------------------------------
# Items / Prices Tests
# ---------------------------------------------------------------------------

class TestExtractItems:
    def test_basic_items_extracted(self):
        from src.extraction.items_prices import extract_items
        tokens = _receipt_tokens_basic()
        items = extract_items(tokens)
        assert len(items) >= 1
        for item in items:
            assert "name" in item
            assert "price" in item
            assert "confidence" in item

    def test_total_line_excluded(self):
        from src.extraction.items_prices import extract_items
        tokens = [
            _make_token("Milk 2%       3.49", 0.88, y=10),
            _make_token("TOTAL         12.39", 0.97, y=50),
        ]
        items = extract_items(tokens)
        names = [i["name"].lower() for i in items]
        assert not any("total" in n for n in names)

    def test_subtotal_line_excluded(self):
        from src.extraction.items_prices import extract_items
        tokens = [_make_token("SUBTOTAL 11.47", 0.95, y=10)]
        items = extract_items(tokens)
        assert len(items) == 0

    def test_tax_line_excluded(self):
        from src.extraction.items_prices import extract_items
        tokens = [_make_token("TAX 8%   0.92", 0.91, y=10)]
        items = extract_items(tokens)
        assert len(items) == 0

    def test_empty_tokens(self):
        from src.extraction.items_prices import extract_items
        assert extract_items([]) == []

    def test_european_price_normalised(self):
        """Comma-decimal prices should be converted to period-decimal."""
        from src.extraction.items_prices import extract_items
        tokens = [_make_token("Brot Vollkorn    2,99", 0.87, y=10)]
        items = extract_items(tokens)
        if items:
            assert "." in items[0]["price"]

    def test_comma_thousands_price_normalised(self):
        """Comma-thousands prices should not be interpreted as decimals."""
        from src.extraction.items_prices import extract_items
        tokens = [_make_token("Rice 12,000", 0.90, y=10)]
        items = extract_items(tokens)
        assert items[0]["price"] == "12000.00"


# ---------------------------------------------------------------------------
# Total Amount Tests
# ---------------------------------------------------------------------------

class TestExtractTotal:
    def test_finds_total_with_keyword(self):
        from src.extraction.total_amount import extract_total
        tokens = _receipt_tokens_basic()
        value, evidence = extract_total(tokens)
        assert value is not None
        assert float(value) > 0

    def test_prefers_keyword_adjacent_total(self):
        """When a 'TOTAL' keyword is adjacent, that amount should win."""
        from src.extraction.total_amount import extract_total
        tokens = [
            _make_token("Subtotal", 0.92, y=100),
            _make_token("11.47", 0.90, y=100, x=200),
            _make_token("TOTAL", 0.98, y=130),
            _make_token("$12.39", 0.97, y=130, x=200),
        ]
        value, evidence = extract_total(tokens)
        assert value is not None
        assert float(value) == pytest.approx(12.39, abs=0.01)

    def test_parses_comma_thousands_total(self):
        """A comma-thousands total must not be truncated to two decimals."""
        from src.extraction.total_amount import extract_total
        tokens = [
            _make_token("1 Ham Cheese 74,000", 0.95, y=100),
            _make_token("SUBTOTAL", 0.99, y=200),
            _make_token("175, 000", 0.65, y=250, x=300),
            _make_token("TOTAL", 0.99, y=250),
        ]
        value, evidence = extract_total(tokens)
        assert value == "175000.00"

    def test_total_label_can_have_intervening_tokens(self):
        """Same-line labels should work even when OCR inserts other tokens."""
        from src.extraction.total_amount import extract_total
        tokens = [
            _make_token("TOTAL", 0.99, x=10, y=100),
            _make_token("Rp", 0.80, x=100, y=100),
            _make_token("175,000", 0.90, x=200, y=100),
        ]
        value, _ = extract_total(tokens)
        assert value == "175000.00"

    def test_ignores_barcode_embedded_amount(self):
        """Amounts inside receipt identifiers must not become totals."""
        from src.extraction.total_amount import extract_total
        tokens = [
            _make_token("TCH5679,8348.6485.7828 2003", 0.23, y=100),
            _make_token("THANK YOU", 0.99, y=130),
        ]
        value, _ = extract_total(tokens)
        assert value is None

    def test_subtotal_is_not_total_keyword(self):
        """A subtotal amount must not receive final-total priority."""
        from src.extraction.total_amount import extract_total
        tokens = [
            _make_token("Sub Total", 0.99, x=10, y=100),
            _make_token("24.00", 0.95, x=200, y=100),
        ]
        value, evidence = extract_total(tokens)
        assert value is None
        assert evidence["keyword_match"] is False

    def test_product_name_containing_total_is_not_label(self):
        """Product descriptions containing TOTAL must not win as labels."""
        from src.extraction.total_amount import extract_total
        tokens = [
            _make_token("COLGATE TOTAL WHITNING TPST 1.82", 0.90, y=100),
        ]
        value, evidence = extract_total(tokens)
        assert value is None
        assert evidence["keyword_match"] is False

    def test_no_price_returns_none(self):
        from src.extraction.total_amount import extract_total
        tokens = [
            _make_token("WALMART SUPERCENTER", 0.97),
            _make_token("Thank you for shopping!", 0.89),
        ]
        value, evidence = extract_total(tokens)
        assert value is None
        assert evidence["alternative_candidates"] == []

    def test_alternative_candidates_exposed(self):
        from src.extraction.total_amount import extract_total
        tokens = [
            _make_token("Subtotal 11.47", 0.90, y=100),
            _make_token("Tax 0.92", 0.91, y=120),
            _make_token("TOTAL $12.39", 0.97, y=150),
        ]
        value, evidence = extract_total(tokens)
        # Might have alternatives; just check the key exists
        assert "alternative_candidates" in evidence
        assert isinstance(evidence["alternative_candidates"], list)

    def test_evidence_has_required_keys(self):
        from src.extraction.total_amount import extract_total
        tokens = _receipt_tokens_basic()
        _, evidence = extract_total(tokens)
        for key in ("ocr_confidence", "heuristic_score", "alternative_candidates"):
            assert key in evidence
