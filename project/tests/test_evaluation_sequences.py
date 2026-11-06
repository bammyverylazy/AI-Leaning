import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from evaluation_sequences import (
    load_attempt_sequence,
    resolve_sequence_path,
    save_attempt_sequence,
)


class EvaluationSequenceTests(unittest.TestCase):
    def test_sequence_round_trip_preserves_arrays_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sequence_dir = root / "data" / "evaluation" / "sequences"
            arrays = {
                "raw_landmarks": np.arange(C.SEQ_LEN * C.N_POINTS * 3, dtype=np.float32)
                .reshape(C.SEQ_LEN, C.N_POINTS, 3),
                "normalized_positions": np.zeros(
                    (C.SEQ_LEN, C.N_POINTS, 3), dtype=np.float32
                ),
                "hand_present": np.ones((C.SEQ_LEN, 2), dtype=np.bool_),
                "model_features": np.ones((C.SEQ_LEN, 590), dtype=np.float32),
            }
            with (
                patch.object(C, "ROOT", root),
                patch.object(C, "TESTING_SEQUENCES", sequence_dir),
            ):
                relative_path = save_attempt_sequence(
                    7,
                    **arrays,
                    metadata={"model_version": "abc123", "attempt_id": 7},
                )
                loaded = load_attempt_sequence(resolve_sequence_path(relative_path))

            self.assertEqual(relative_path, "data/evaluation/sequences/7.npz")
            for name, value in arrays.items():
                np.testing.assert_array_equal(loaded[name], value)
            self.assertEqual(
                loaded["metadata"], {"model_version": "abc123", "attempt_id": 7}
            )

    def test_sequence_path_cannot_escape_testing_storage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sequence_dir = root / "data" / "evaluation" / "sequences"
            outside_file = root / "outside.npz"
            outside_file.touch()
            with (
                patch.object(C, "ROOT", root),
                patch.object(C, "TESTING_SEQUENCES", sequence_dir),
                self.assertRaisesRegex(ValueError, "outside Testing Mode storage"),
            ):
                resolve_sequence_path("outside.npz")


if __name__ == "__main__":
    unittest.main()
