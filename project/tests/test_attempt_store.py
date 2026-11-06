import sqlite3
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from attempt_store import RecognitionAttemptStore


class RecognitionAttemptStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = RecognitionAttemptStore(Path(self.temp_dir.name) / "attempts.sqlite3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_attempt(self, target="AXE1", predicted="DOG1"):
        return self.store.create_attempt(
            target_label=target,
            demo_video_id="demo-axe",
            demo_video_version="1",
            demo_watched=True,
            demo_watch_duration=2.5,
            predicted_label=predicted,
            confidence=0.8,
        )

    def test_new_attempt_keeps_target_prediction_and_ground_truth_separate(self):
        attempt_id = self.create_attempt()

        connection = sqlite3.connect(self.store.path)
        try:
            row = connection.execute(
                """
                SELECT target_label, predicted_label, ground_truth_label,
                       correctness_status, demo_watched, demo_watch_duration
                FROM recognition_attempts WHERE attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
        finally:
            connection.close()

        self.assertEqual(row, ("AXE1", "DOG1", None, "unverified", 1, 2.5))

    def test_attempt_records_inference_provenance(self):
        attempt_id = self.store.create_attempt(
            target_label="WHATFOR1",
            demo_video_id="demo",
            demo_video_version="2",
            demo_watched=True,
            demo_watch_duration=1.25,
            predicted_label="WHAT FOR",
            confidence=0.7,
            session_id="session-1",
            original_model_label="WHATFOR1",
            canonical_label="WHAT FOR",
            class_index=49,
            top_k_predictions=[
                {
                    "canonical_label": "WHAT FOR",
                    "original_model_label": "WHATFOR1",
                    "class_index": 49,
                    "confidence": 0.7,
                }
            ],
            model_version="sha256:abc",
            preprocessing_version="pre-v1",
            inference_latency_ms=12.5,
        )
        connection = sqlite3.connect(self.store.path)
        try:
            row = connection.execute(
                """
                SELECT session_id, target_label, predicted_label,
                       original_model_label, canonical_label, class_index,
                       top_k_predictions, model_version, preprocessing_version,
                       inference_latency_ms, timestamp
                FROM recognition_attempts WHERE attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
        finally:
            connection.close()

        self.assertEqual(row[:10], (
            "session-1", "WHATFOR1", "WHAT FOR", "WHATFOR1", "WHAT FOR",
            49, (
                '[{"canonical_label": "WHAT FOR", "original_model_label": "WHATFOR1", '
                '"class_index": 49, "confidence": 0.7}]'
            ),
            "sha256:abc", "pre-v1", 12.5,
        ))
        self.assertTrue(row[10])

    def test_review_compares_prediction_to_ground_truth_not_target(self):
        attempt_id = self.create_attempt(target="AXE1", predicted="DOG1")

        status = self.store.verify_ground_truth(attempt_id, "DOG1", "reviewer")

        self.assertEqual(status, "correct")
        self.assertEqual(self.store.pending_attempts(), [])
        connection = sqlite3.connect(self.store.path)
        try:
            row = connection.execute(
                """
                SELECT target_label, predicted_label, ground_truth_label, correctness_status
                FROM recognition_attempts WHERE attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(row, ("AXE1", "DOG1", "DOG1", "correct"))

    def test_review_marks_prediction_wrong_when_it_differs_from_ground_truth(self):
        attempt_id = self.create_attempt(target="AXE1", predicted="DOG1")

        status = self.store.verify_ground_truth(attempt_id, "AXE1", "reviewer")

        self.assertEqual(status, "incorrect")

    def test_review_compares_confirmed_canonical_aliases_as_same_label(self):
        attempt_id = self.create_attempt(predicted="WHAT FOR")

        status = self.store.verify_ground_truth(attempt_id, "WHATFOR1", "reviewer")

        self.assertEqual(status, "correct")

    def test_review_requires_reviewer_and_cannot_overwrite_verified_ground_truth(self):
        attempt_id = self.create_attempt()
        with self.assertRaises(ValueError):
            self.store.verify_ground_truth(attempt_id, "DOG1", " ")

        self.store.verify_ground_truth(attempt_id, "DOG1", "reviewer")
        with self.assertRaises(ValueError):
            self.store.verify_ground_truth(attempt_id, "AXE1", "reviewer")

    def test_correction_is_audited_and_keeps_original_prediction(self):
        attempt_id = self.create_attempt(predicted="DOG1")
        self.store.verify_ground_truth(attempt_id, "AXE1", "reviewer")
        with self.assertRaises(ValueError):
            self.store.correct_ground_truth(attempt_id, "DOG1", "reviewer", "")

        status = self.store.correct_ground_truth(
            attempt_id, "DOG1", "reviewer-2", "independent video review"
        )
        connection = sqlite3.connect(self.store.path)
        try:
            attempt = connection.execute(
                """
                SELECT predicted_label, ground_truth_label, correctness_status
                FROM recognition_attempts WHERE attempt_id = ?
                """,
                (attempt_id,),
            ).fetchone()
            history = connection.execute(
                """
                SELECT action, previous_ground_truth_label, new_ground_truth_label,
                       previous_correctness_status, new_correctness_status,
                       reviewer, reason
                FROM recognition_attempt_reviews
                WHERE attempt_id = ? ORDER BY review_id
                """,
                (attempt_id,),
            ).fetchall()
        finally:
            connection.close()

        self.assertEqual(status, "correct")
        self.assertEqual(attempt, ("DOG1", "DOG1", "correct"))
        self.assertEqual(history[1][:], (
            "correct", "AXE1", "DOG1", "incorrect", "correct",
            "reviewer-2", "independent video review",
        ))

    def test_legacy_database_is_migrated_without_losing_attempts(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_path)
        connection.execute(
            """
            CREATE TABLE recognition_attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT '2025-01-01T00:00:00Z',
                target_label TEXT NOT NULL,
                demo_video_id TEXT NOT NULL,
                demo_video_version TEXT NOT NULL,
                demo_watched INTEGER NOT NULL,
                demo_watch_duration REAL NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                ground_truth_label TEXT,
                correctness_status TEXT NOT NULL DEFAULT 'unverified',
                verified_by TEXT,
                verified_at TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO recognition_attempts (
                target_label, demo_video_id, demo_video_version, demo_watched,
                demo_watch_duration, predicted_label, confidence
            ) VALUES ('AXE1', 'demo', '1', 0, 0, 'DOG1', 0.6)
            """
        )
        connection.commit()
        connection.close()

        migrated = RecognitionAttemptStore(legacy_path)
        rows = migrated.pending_attempts()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["original_model_label"], "DOG1")
        self.assertEqual(rows[0]["canonical_label"], "DOG1")
        self.assertEqual(rows[0]["session_id"], "legacy")

    def test_evaluation_session_and_attempt_keep_full_provenance(self):
        session_id = self.store.start_session(
            "evaluation-session",
            model_version="model-sha",
            vocabulary_version="vocab-sha",
            configuration={"confidence_threshold": 0.45},
            signer_id="signer-01",
            environment={"camera_index": 0, "lighting": "office"},
            notes="first evaluation",
            model_checkpoint="models/test.pt",
            application_version="test-build",
            model_id="test-model",
        )
        attempt_id = self.store.begin_attempt(
            session_id=session_id,
            target_label="AXE1",
            model_version="model-sha",
            preprocessing_version="pre-v1",
        )

        self.store.finish_attempt(
            attempt_id,
            status="completed",
            predicted_label="DOG1",
            original_model_label="DOG1",
            canonical_label_value="DOG (variant 1)",
            class_index=1,
            confidence=0.8,
            top_k_predictions=[
                {
                    "canonical_label": "DOG (variant 1)",
                    "original_model_label": "DOG1",
                    "class_index": 1,
                    "confidence": 0.8,
                }
            ],
            sequence_path="data/evaluation/sequences/1.npz",
            sequence_length=64,
            feature_dimension=590,
            frame_count=128,
        )
        session = self.store.get_session(session_id)
        attempt = self.store.get_attempt(attempt_id)

        self.assertEqual(session["signer_id"], "signer-01")
        self.assertEqual(session["environment_json"], '{"camera_index": 0, "lighting": "office"}')
        self.assertEqual(session["notes"], "first evaluation")
        self.assertEqual(attempt["target_label"], "AXE1")
        self.assertIsNone(attempt["ground_truth_label"])
        self.assertEqual(attempt["correctness_status"], "unverified")
        self.assertEqual(attempt["model_id"], "test-model")
        self.assertEqual(attempt["vocabulary_version"], "vocab-sha")
        self.assertEqual(attempt["sequence_path"], "data/evaluation/sequences/1.npz")
        self.assertEqual(attempt["sequence_length"], 64)
        self.assertEqual(attempt["feature_dimension"], 590)
        self.assertEqual(attempt["frame_count"], 128)

    def test_session_lifecycle_attempt_trace_and_no_prediction_status(self):
        session_id = self.store.start_session(
            "session-1",
            model_version="model-sha",
            vocabulary_version="vocab-sha",
            configuration={"threshold": 0.45},
        )
        attempt_id = self.store.begin_attempt(
            session_id=session_id,
            target_label="AXE1",
            model_version="model-sha",
            preprocessing_version="pre-v1",
        )
        attempt = self.store.get_attempt(attempt_id)
        self.assertEqual(attempt["status"], "in_progress")
        self.assertEqual(attempt["ground_truth_label"], None)
        self.assertEqual(attempt["demo_video_id"], "")

        self.assertEqual(
            self.store.finish_attempt(
                attempt_id,
                status="no_prediction",
                warning="No stable sign was confirmed",
            ),
            "no_prediction",
        )
        self.assertEqual(
            self.store.get_attempt(attempt_id)["status"],
            "no_prediction",
        )
        self.assertEqual(self.store.pending_attempts(), [])
        self.assertEqual(
            [row["event_type"] for row in self.store.attempt_events(attempt_id)],
            [
                "session_started",
                "target_selected",
                "attempt_started",
                "recognition_started",
                "attempt_completed",
                "attempt_saved",
            ],
        )
        self.assertEqual(self.store.end_session(session_id), "completed")
        self.assertEqual(self.store.get_session(session_id)["status"], "completed")

    def test_attempt_cancellation_and_interrupted_session_are_explicit(self):
        session_id = self.store.start_session(
            "session-interrupted",
            model_version="model",
            vocabulary_version="vocab",
            configuration={},
        )
        attempt_id = self.store.begin_attempt(
            session_id=session_id,
            target_label="AXE1",
            model_version="model",
            preprocessing_version="pre-v1",
        )
        self.store.finish_attempt(
            attempt_id,
            status="cancelled",
            error="user cancelled",
        )
        self.assertEqual(self.store.get_attempt(attempt_id)["status"], "cancelled")
        self.assertEqual(self.store.end_session(session_id, "interrupted"), "interrupted")

    def test_incomplete_attempt_cannot_be_verified(self):
        session_id = self.store.start_session(
            "session-pending",
            model_version="model",
            vocabulary_version="vocab",
            configuration={},
        )
        attempt_id = self.store.begin_attempt(
            session_id=session_id,
            target_label="AXE1",
            model_version="model",
            preprocessing_version="pre-v1",
        )
        with self.assertRaisesRegex(ValueError, "completed attempts"):
            self.store.verify_ground_truth(attempt_id, "AXE1", "reviewer")


if __name__ == "__main__":
    unittest.main()
