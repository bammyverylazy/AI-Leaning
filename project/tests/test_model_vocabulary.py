import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_vocabulary import (
    build_class_mapping,
    checkpoint_vocabulary,
    validate_vocabulary,
)


class ModelVocabularyTests(unittest.TestCase):
    def test_builds_stable_ordered_mapping_and_reads_it_back(self):
        labels = ["AXE1", "DOG1"]
        mapping = build_class_mapping(labels)

        self.assertEqual(
            mapping,
            [
                {
                    "class_index": 0,
                    "label": "AXE1",
                    "canonical_label": "AXE (variant 1)",
                },
                {
                    "class_index": 1,
                    "label": "DOG1",
                    "canonical_label": "DOG (variant 1)",
                },
            ],
        )
        self.assertEqual(
            checkpoint_vocabulary(
                {"vocab": labels, "class_mapping": mapping}
            ),
            labels,
        )

    def test_rejects_duplicate_raw_and_canonical_labels(self):
        with self.assertRaisesRegex(ValueError, "duplicate class"):
            validate_vocabulary(["AXE1", "AXE1"])
        with self.assertRaisesRegex(ValueError, "duplicate canonical"):
            validate_vocabulary(["WHATFOR1", "WHATFOR2"])

    def test_rejects_noncontiguous_or_mismatched_checkpoint_mapping(self):
        labels = ["AXE1", "DOG1"]
        mapping = build_class_mapping(labels)
        mapping[1]["class_index"] = 0
        with self.assertRaisesRegex(ValueError, "unique and contiguous"):
            checkpoint_vocabulary({"vocab": labels, "class_mapping": mapping})

        mapping = build_class_mapping(labels)
        mapping[0]["canonical_label"] = "DOG (variant 1)"
        with self.assertRaisesRegex(ValueError, "does not match"):
            checkpoint_vocabulary({"vocab": labels, "class_mapping": mapping})

    def test_supports_legacy_checkpoints_with_ordered_vocab(self):
        self.assertEqual(
            checkpoint_vocabulary({"vocab": ["AXE1", "DOG1"]}),
            ["AXE1", "DOG1"],
        )


if __name__ == "__main__":
    unittest.main()
