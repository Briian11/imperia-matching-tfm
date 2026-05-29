from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imperia_matching_tfm.matching.hybrid_m4 import score_property_for_client_m4
from imperia_matching_tfm.matching.semantic import (
    build_client_text,
    build_property_text,
    cosine_similarity,
    semantic_score,
)
from imperia_matching_tfm.models import (
    BuyerPreference,
    Client,
    Operation,
    Property,
    PropertyType,
)


def _client() -> Client:
    pref = BuyerPreference(
        operation=Operation.BUY,
        property_types=[PropertyType.HOUSE],
        zones=["Ruzafa"],
        cities=["Valencia"],
        price_max=400000,
        features=["terraza"],
        notes="Busco casa luminosa cerca del centro de Valencia",
    )
    return Client(id="c_test", name="Test", preference=pref)


def _property() -> Property:
    return Property(
        id="p_test",
        title="Casa luminosa en Ruzafa",
        operation=Operation.BUY,
        property_type=PropertyType.HOUSE,
        price=350000,
        city="Valencia",
        zone="Ruzafa",
        features=["terrace"],
        description="Vivienda con mucha luz natural en pleno centro",
    )


class SemanticTextBuildersTest(unittest.TestCase):
    def test_client_text_includes_notes_and_zones(self):
        text = build_client_text(_client())
        self.assertIn("luminosa", text)
        self.assertIn("Ruzafa", text)

    def test_property_text_includes_title_and_features(self):
        text = build_property_text(_property())
        self.assertIn("Ruzafa", text)
        self.assertIn("terrace", text)


class CosineSimilarityTest(unittest.TestCase):
    def test_identical_vectors_return_one(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal_vectors_return_zero(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_empty_vectors_return_zero(self):
        self.assertEqual(cosine_similarity([], []), 0.0)


class M4HybridTest(unittest.TestCase):
    def test_m4_keeps_dealbreaker_when_operation_mismatches(self):
        client = _client()
        prop = _property()
        prop.operation = Operation.RENT
        cache = {
            "properties": {"p_test": [1.0, 0.0, 0.0]},
            "clients": {"c_test": [1.0, 0.0, 0.0]},
        }
        match = score_property_for_client_m4(client, prop, cache, alpha=0.5)
        self.assertEqual(match.score, 0)

    def test_m4_adds_semantic_criterion(self):
        client = _client()
        prop = _property()
        cache = {
            "properties": {"p_test": [1.0, 0.0, 0.0]},
            "clients": {"c_test": [1.0, 0.0, 0.0]},
        }
        match = score_property_for_client_m4(client, prop, cache, alpha=0.7)
        criterion_names = [c.name for c in match.criteria]
        self.assertIn("semantic", criterion_names)

    def test_semantic_score_in_unit_range(self):
        cache = {
            "properties": {"p_test": [0.5, 0.5, 0.0]},
            "clients": {"c_test": [0.5, 0.5, 0.0]},
        }
        score = semantic_score("c_test", "p_test", cache)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
