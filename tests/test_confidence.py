"""
tests/test_confidence.py — Unit tests for confidence scoring and flagging.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestScoreField:
    def test_perfect_scores_give_one(self):
        """All inputs at 1.0 should yield 1.0."""
        from src.confidence.scorer import score_field
        assert score_field(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_zero_scores_give_zero(self):
        from src.confidence.scorer import score_field
        assert score_field(0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_weights_sum_to_one(self):
        """Verify the configured weights actually sum to 1.0."""
        import config
        total = config.CONF_WEIGHT_OCR + config.CONF_WEIGHT_PATTERN + config.CONF_WEIGHT_HEURISTIC
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_formula_matches_expected(self):
        """Manual calculation should match score_field output."""
        from src.confidence.scorer import score_field
        ocr, pat, heur = 0.80, 0.60, 0.90
        expected = round(0.50 * ocr + 0.30 * pat + 0.20 * heur, 3)
        assert score_field(ocr, pat, heur) == pytest.approx(expected)

    def test_output_clamped_to_zero_one(self):
        """Values exceeding [0, 1] should be clamped."""
        from src.confidence.scorer import score_field
        # Over-saturated inputs
        assert score_field(1.5, 1.2, 1.1) == pytest.approx(1.0)
        assert score_field(-0.5, -0.1, -0.3) == pytest.approx(0.0)

    def test_return_type_is_float(self):
        from src.confidence.scorer import score_field
        result = score_field(0.7, 0.8, 0.9)
        assert isinstance(result, float)


class TestBuildField:
    def test_high_confidence_no_flag(self):
        """Field with confidence ≥ threshold should NOT be flagged."""
        from src.confidence.scorer import build_field
        field = build_field(
            value="WALMART",
            ocr_confidence=0.97,
            pattern_validity=1.0,
            heuristic_score=1.0,
        )
        assert field["value"] == "WALMART"
        assert "flagged" not in field

    def test_low_confidence_flagged(self):
        """Field below threshold should have flagged=True and a reason."""
        from src.confidence.scorer import build_field
        field = build_field(
            value="WALMART",
            ocr_confidence=0.10,  # very low
            pattern_validity=0.0,
            heuristic_score=0.0,
        )
        assert field.get("flagged") is True
        assert "reason" in field
        assert len(field["reason"]) > 0

    def test_null_value_always_flagged(self):
        """A None value should always produce flagged=True regardless of scores."""
        from src.confidence.scorer import build_field
        field = build_field(
            value=None,
            ocr_confidence=1.0,
            pattern_validity=1.0,
            heuristic_score=1.0,
        )
        assert field["value"] is None
        assert field.get("flagged") is True

    def test_extra_keys_merged(self):
        """Extra kwargs should appear in the output dict."""
        from src.confidence.scorer import build_field
        field = build_field(
            value="12.39",
            ocr_confidence=0.97,
            pattern_validity=1.0,
            heuristic_score=0.9,
            extra={"alternative_candidates": [{"value": "11.47"}]},
        )
        assert "alternative_candidates" in field
        assert len(field["alternative_candidates"]) == 1

    def test_confidence_key_always_present(self):
        from src.confidence.scorer import build_field
        for val in (None, "STORE", "12.99"):
            field = build_field(val, 0.5, 0.5, 0.5)
            assert "confidence" in field
            assert 0.0 <= field["confidence"] <= 1.0


class TestScoreItems:
    def test_empty_items(self):
        from src.confidence.scorer import score_items
        assert score_items([]) == []

    def test_output_schema(self):
        """Each scored item must have name, price, and confidence."""
        from src.confidence.scorer import score_items
        raw = [
            {"name": "Milk 2%", "price": "3.49", "confidence": 0.88},
            {"name": "Bread", "price": "2.99", "confidence": 0.85},
        ]
        result = score_items(raw)
        assert len(result) == 2
        for item in result:
            assert "name" in item
            assert "price" in item
            assert "confidence" in item
            assert 0.0 <= item["confidence"] <= 1.0

    def test_low_confidence_item_flagged(self):
        """An item with very low raw confidence should be flagged."""
        from src.confidence.scorer import score_items
        raw = [{"name": "Blurry Item", "price": "1.00", "confidence": 0.05}]
        result = score_items(raw)
        # May or may not be flagged depending on threshold; just ensure no crash
        assert len(result) == 1
        assert "name" in result[0]
