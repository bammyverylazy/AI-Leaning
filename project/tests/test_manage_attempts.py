import contextlib
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from attempt_store import RecognitionAttemptStore
from dataset import build_features, preprocess_sequence
from evaluation_sequences import save_attempt_sequence
from label_display import aggregate_predictions
from manage_attempts import main, replay_attempt
from model import SeqClassifier


class ManageAttemptsTests(unittest.TestCase):
    def test_replay_reproduces_saved_model_input_with_matching_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sequence_dir = root / "data" / "evaluation" / "sequences"
            checkpoint_path = root / "test-model.pt"
            rng = np.random.default_rng(42)
            landmarks = rng.normal(
                size=(C.SEQ_LEN, C.N_POINTS, 3)
            ).astype(np.float32)
            landmarks[:, C.L_SH, :] = [0.35, 0.4, 0.02]
            landmarks[:, C.R_SH, :] = [0.65, 0.4, 0.02]
            positions, present = preprocess_sequence(landmarks)
            features = build_features(positions, present)
            model_args = {
                "hidden": 4,
                "dropout": 0.0,
                "kind": "gru",
            }
            model = SeqClassifier(
                features.shape[1], 2, model_args["hidden"], 2,
                model_args["dropout"], model_args["kind"],
            )
            checkpoint = {
                "feature_dim": features.shape[1],
                "vocab": ["SIGN_A", "SIGN_B"],
                "args": model_args,
                "model": model.state_dict(),
            }
            with (
                patch.object(C, "ROOT", root),
                patch.object(C, "TESTING_SEQUENCES", sequence_dir),
            ):
                torch.save(checkpoint, checkpoint_path)
                sequence_path = save_attempt_sequence(
                    1,
                    raw_landmarks=landmarks,
                    normalized_positions=positions,
                    hand_present=present,
                    model_features=features,
                    metadata={"attempt_id": 1},
                )
                with torch.no_grad():
                    probabilities = torch.softmax(
                        model(torch.from_numpy(features)[None])[0], dim=0
                    )
                expected = aggregate_predictions(
                    probabilities, checkpoint["vocab"], top_k=3
                )
                attempt = {
                    "attempt_id": 1,
                    "sequence_path": sequence_path,
                    "status": "completed",
                    "original_model_label": expected[0].original_model_label,
                    "model_checkpoint": str(checkpoint_path),
                    "model_version": hashlib.sha256(
                        checkpoint_path.read_bytes()
                    ).hexdigest(),
                    "feature_dimension": features.shape[1],
                    "sequence_length": C.SEQ_LEN,
                    "top_k_predictions": json.dumps(
                        [
                            {
                                "class_index": prediction.class_index,
                                "original_model_label": prediction.original_model_label,
                            }
                            for prediction in expected
                        ]
                    ),
                    "canonical_label": expected[0].canonical_label,
                    "confidence": expected[0].confidence,
                }
                result = replay_attempt(attempt)

            self.assertTrue(result["preprocessing_matches"])
            self.assertTrue(result["prediction_matches"])
            self.assertEqual(
                result["predictions"][0].canonical_label,
                expected[0].canonical_label,
            )
            missing_top_k = replay_attempt(
                {**attempt, "top_k_predictions": "[]"}
            )
            self.assertFalse(missing_top_k["prediction_matches"])

    def test_export_contains_only_verified_completed_attempts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database = root / "attempts.sqlite3"
            output = root / "verified.csv"
            store = RecognitionAttemptStore(database)
            session_id = store.start_session(
                "export-session",
                model_version="model-sha",
                vocabulary_version="vocab-sha",
                configuration={},
            )
            for index in range(2):
                attempt_id = store.begin_attempt(
                    session_id=session_id,
                    target_label=f"SIGN{index}",
                    model_version="model-sha",
                    preprocessing_version="pre-v1",
                )
                store.finish_attempt(
                    attempt_id,
                    status="completed",
                    predicted_label=f"SIGN{index}",
                    original_model_label=f"SIGN{index}",
                    canonical_label_value=f"SIGN{index}",
                    confidence=0.9,
                )
                if index == 0:
                    store.verify_ground_truth(
                        attempt_id, "SIGN0", "independent-reviewer"
                    )

            output_capture = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "manage_attempts.py",
                        "--db",
                        str(database),
                        "export",
                        "--output",
                        str(output),
                    ],
                ),
                contextlib.redirect_stdout(output_capture),
            ):
                main()

            with output.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["attempt_id"] for row in rows], ["1"])
            self.assertEqual(rows[0]["ground_truth_label"], "SIGN0")
            self.assertIn("Exported 1 independently verified attempts", output_capture.getvalue())

            inspect_capture = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    ["manage_attempts.py", "--db", str(database), "inspect", "1"],
                ),
                contextlib.redirect_stdout(inspect_capture),
            ):
                main()
            self.assertIn("Ground truth: SIGN0", inspect_capture.getvalue())
            self.assertIn("Feature dimension:", inspect_capture.getvalue())


if __name__ == "__main__":
    unittest.main()
