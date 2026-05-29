import unittest

from imperia_matching_tfm.evaluation.ranking import ndcg_at_k, precision_at_k, recall_at_k


class RankingMetricsTest(unittest.TestCase):
    def test_precision_at_k(self) -> None:
        self.assertEqual(precision_at_k([1, 0, 1], 2), 0.5)

    def test_recall_at_k(self) -> None:
        self.assertEqual(recall_at_k([1, 0, 1], total_relevant=2, k=3), 1.0)

    def test_ndcg_at_k(self) -> None:
        self.assertEqual(ndcg_at_k([3, 2, 0], 3), 1.0)


if __name__ == "__main__":
    unittest.main()
