import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from label_display import aggregate_predictions, canonical_label


class LabelDisplayTests(unittest.TestCase):
    def test_only_confirmed_aliases_are_merged(self):
        self.assertEqual(canonical_label("WHATFOR1"), "WHAT FOR")
        self.assertEqual(canonical_label("WHATFOR2"), "WHAT FOR")
        self.assertEqual(canonical_label("CHRISTMAS1"), "CHRISTMAS (variant 1)")
        self.assertEqual(canonical_label("CHRISTMAS2"), "CHRISTMAS (variant 2)")
        self.assertEqual(
            canonical_label("CHRISTMAS (variant 1)"),
            canonical_label("CHRISTMAS1"),
        )

    def test_probability_aggregation_preserves_original_class_provenance(self):
        predictions = aggregate_predictions(
            torch.tensor([0.35, 0.30, 0.35]),
            ["WHATFOR1", "WHATFOR2", "CHRISTMAS1"],
            top_k=3,
        )

        self.assertEqual(predictions[0].canonical_label, "WHAT FOR")
        self.assertAlmostEqual(predictions[0].confidence, 0.65)
        self.assertEqual(predictions[0].original_model_label, "WHATFOR1")
        self.assertEqual(predictions[0].class_index, 0)
        self.assertEqual(predictions[1].canonical_label, "CHRISTMAS (variant 1)")
        self.assertEqual(predictions[1].class_index, 2)

    def test_probability_count_and_top_k_are_validated(self):
        with self.assertRaises(ValueError):
            aggregate_predictions(torch.tensor([0.5]), ["AXE1", "DOG1"])
        with self.assertRaises(ValueError):
            aggregate_predictions(torch.tensor([0.5]), ["AXE1"], top_k=0)


if __name__ == "__main__":
    unittest.main()
