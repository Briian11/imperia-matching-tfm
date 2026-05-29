import unittest

from imperia_matching_tfm.matching.baseline import score_property_for_client
from imperia_matching_tfm.models import BuyerPreference, Client, Operation, Property, PropertyType


class BaselineScoringTest(unittest.TestCase):
    def test_operation_mismatch_is_dealbreaker(self) -> None:
        client = Client(
            id="c1",
            name="Cliente",
            preference=BuyerPreference(operation=Operation.BUY),
        )
        property_ = Property(
            id="p1",
            title="Piso alquiler",
            operation=Operation.RENT,
            property_type=PropertyType.FLAT,
            price=1200,
            city="Madrid",
        )

        result = score_property_for_client(client, property_)

        self.assertEqual(result.score, 0)
        self.assertEqual(result.criteria[0].name, "operation")
        self.assertFalse(result.criteria[0].matched)

    def test_good_match_scores_high(self) -> None:
        client = Client(
            id="c1",
            name="Cliente",
            preference=BuyerPreference(
                operation=Operation.BUY,
                property_types=[PropertyType.FLAT],
                cities=["Madrid"],
                price_max=350000,
                bedrooms_min=2,
                bathrooms_min=1,
                surface_min_m2=70,
                features=["terraza", "garaje"],
            ),
        )
        property_ = Property(
            id="p1",
            title="Piso con terraza",
            operation=Operation.BUY,
            property_type=PropertyType.FLAT,
            price=330000,
            city="Madrid",
            bedrooms=3,
            bathrooms=2,
            surface_m2=82,
            features=["terrace", "garage", "elevator"],
        )

        result = score_property_for_client(client, property_)

        self.assertGreaterEqual(result.score, 90)


if __name__ == "__main__":
    unittest.main()
