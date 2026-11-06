import csv
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from evaluation_sequences import load_attempt_sequence
from dataset import build_features, preprocess_sequence
from self_recorded_store import SelfRecordedSampleStore, validate_recording_arrays


class SelfRecordedStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "self_recorded"
        self.config_patch = patch.object(C, "SELF_RECORDING", self.root)
        self.db_patch = patch.object(
            C, "SELF_RECORDING_DB", self.root / "samples.sqlite3"
        )
        self.config_patch.start()
        self.db_patch.start()
        self.store = SelfRecordedSampleStore()

    def tearDown(self):
        self.db_patch.stop()
        self.config_patch.stop()
        self.temp_dir.cleanup()

    def _create_sample(
        self,
        sample_id,
        *,
        signer_id="S001",
        session_id="session-1",
        target_label="AXE1",
        dominant_hand="UNKNOWN",
        camera_index=0,
        camera_angle="front",
        raw_video_path="",
        offset=0.0,
    ):
        raw = np.full((16, C.N_POINTS, 3), np.nan, dtype=np.float32)
        raw[:, C.LH, :] = (0.4 + offset, 0.5, 0.0)
        raw[:, C.RH, :] = (0.6, 0.5, 0.0)
        positions, present = preprocess_sequence(raw)
        features = build_features(positions, present)
        return self.store.create_pending(
            sample_id=sample_id,
            target_label=target_label,
            signer_id=signer_id,
            session_id=session_id,
            dominant_hand=dominant_hand,
            camera_source=f"opencv-index:{camera_index}",
            camera_index=camera_index,
            camera_angle=camera_angle,
            distance="normal",
            lighting="normal",
            signing_speed="normal",
            notes="test sample",
            duration_seconds=1.2,
            original_frame_count=16,
            raw_landmarks=raw,
            normalized_positions=positions,
            hand_present=present,
            model_features=features,
            model_version="checkpoint-sha",
            raw_video_path=raw_video_path,
        )

    def test_saves_and_reloads_sequence_metadata_then_accepts_or_rejects(self):
        video_path = self.root / "raw" / "accepted-1.avi"
        video_path.write_bytes(b"optional video")
        saved = self._create_sample(
            "accepted-1",
            camera_index=2,
            camera_angle="slightly left",
            raw_video_path=str(video_path),
        )
        sample = self.store.get("accepted-1")
        loaded = load_attempt_sequence(Path(saved["sequence_path"]))
        self.assertEqual(loaded["metadata"]["session_id"], "session-1")

        self.assertEqual(sample["status"], "pending_review")
        self.assertIsNone(sample["ground_truth_label"])
        self.assertEqual(sample["target_label"], "AXE1")
        self.assertEqual(sample["signer_id"], "S001")
        self.assertEqual(sample["session_id"], "session-1")
        self.assertEqual(sample["dominant_hand"], "UNKNOWN")
        self.assertEqual(sample["recording_status"], "PENDING_REVIEW")
        self.assertEqual(sample["verification_status"], "UNVERIFIED")
        self.assertEqual(sample["quality_status"], "PASSED")
        self.assertEqual(sample["pose_frame_count"], 0)
        self.assertEqual(sample["shoulder_frame_count"], 0)
        self.assertEqual(sample["camera_index"], 2)
        self.assertEqual(sample["camera_angle"], "slightly left")
        self.assertTrue(sample["raw_video_path"].endswith("accepted-1.avi"))
        self.assertEqual(sample["original_frame_count"], 16)
        self.assertEqual(sample["sequence_length"], C.SEQ_LEN)
        self.assertEqual(sample["feature_dimension"], 590)
        self.assertIsNotNone(sample["recorded_at"])
        self.assertEqual(loaded["model_features"].shape, (64, 590))
        self.assertEqual(loaded["metadata"]["session_id"], "session-1")
        self.assertEqual(loaded["metadata"]["ground_truth_label"], None)
        self.assertEqual(loaded["metadata"]["verification_status"], "UNVERIFIED")
        self.assertEqual(loaded["metadata"]["sequence_path"], sample["sequence_path"])
        accepted = self.store.review("accepted-1", "accepted", "clear sample")
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(accepted["recording_status"], "ACCEPTED_UNVERIFIED")
        self.assertEqual(accepted["verification_status"], "UNVERIFIED")
        self.assertIsNone(accepted["ground_truth_label"])
        accepted_sequence = load_attempt_sequence(Path(accepted["sequence_path"]))
        self.assertEqual(
            accepted_sequence["metadata"]["recording_status"],
            "ACCEPTED_UNVERIFIED",
        )
        self.assertEqual(Path(accepted["sequence_path"]).parent.name, "accepted")
        self.assertEqual(Path(accepted["raw_video_path"]).parent.name, "accepted")
        self.assertTrue(Path(accepted["raw_video_path"]).is_file())

        rejected_saved = self._create_sample("rejected-1", offset=0.1)
        self.assertEqual(self.store.get("rejected-1")["raw_video_path"], "")
        rejected = self.store.review(
            "rejected-1", "rejected", "hand occluded in review"
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(
            rejected["rejection_reason"], "hand occluded in review"
        )
        self.assertEqual(Path(rejected["sequence_path"]).parent.name, "rejected")
        self.assertTrue(Path(rejected_saved["sequence_path"]).exists() is False)

    def test_rejection_requires_an_explanatory_reason(self):
        self._create_sample("no-reason")
        with self.assertRaisesRegex(ValueError, "clear rejection reason"):
            self.store.review("no-reason", "rejected")
        self.assertEqual(self.store.get("no-reason")["status"], "pending_review")

    def test_missing_sequence_cannot_be_accepted_but_can_be_rejected(self):
        created = self._create_sample("missing-sequence")
        Path(created["sequence_path"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "sequence is missing"):
            self.store.review("missing-sequence", "accepted")
        rejected = self.store.review(
            "missing-sequence",
            "rejected",
            rejection_reason="Saved landmark sequence is missing.",
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["recording_status"], "REJECTED")
        self.assertEqual(
            rejected["rejection_reason"], "Saved landmark sequence is missing."
        )

    def test_rejects_short_or_missing_hand_sequences(self):
        with self.assertRaisesRegex(ValueError, "sample ID"):
            self.store.create_pending(
                sample_id="../outside",
                target_label="AXE1",
                signer_id="S001",
                session_id="session-1",
                camera_source="opencv-index:0",
                duration_seconds=1.0,
                original_frame_count=16,
                raw_landmarks=np.zeros((16, C.N_POINTS, 3), dtype=np.float32),
                normalized_positions=np.zeros(
                    (C.SEQ_LEN, C.N_POINTS, 3), dtype=np.float32
                ),
                hand_present=np.ones((C.SEQ_LEN, 2), dtype=np.float32),
                model_features=np.zeros((C.SEQ_LEN, 590), dtype=np.float32),
                model_version="checkpoint-sha",
            )
        raw = np.full((7, C.N_POINTS, 3), np.nan, dtype=np.float32)
        raw[:, C.LH, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "Sequence too short"):
            self.store.create_pending(
                sample_id="too-short",
                target_label="AXE1",
                signer_id="S001",
                session_id="session-1",
                camera_source="opencv-index:0",
                duration_seconds=0.2,
                original_frame_count=7,
                raw_landmarks=raw,
                normalized_positions=np.zeros(
                    (C.SEQ_LEN, C.N_POINTS, 3), dtype=np.float32
                ),
                hand_present=np.ones((C.SEQ_LEN, 2), dtype=np.float32),
                model_features=np.zeros((C.SEQ_LEN, 590), dtype=np.float32),
                model_version="checkpoint-sha",
            )
        raw = np.full((16, C.N_POINTS, 3), np.nan, dtype=np.float32)
        positions, present = preprocess_sequence(raw)
        features = build_features(positions, present)
        with self.assertRaisesRegex(ValueError, "Hand landmarks"):
            self.store.create_pending(
                sample_id="no-hands",
                target_label="AXE1",
                signer_id="S001",
                session_id="session-1",
                camera_source="opencv-index:0",
                duration_seconds=1.0,
                original_frame_count=16,
                raw_landmarks=raw,
                normalized_positions=positions,
                hand_present=present,
                model_features=features,
                model_version="checkpoint-sha",
            )

    def test_rejects_partially_missing_landmark_coordinates(self):
        raw = np.full((16, C.N_POINTS, 3), np.nan, dtype=np.float32)
        raw[:, C.LH, :] = (0.4, 0.5, 0.0)
        raw[:, C.RH, :] = (0.6, 0.5, 0.0)
        positions, present = preprocess_sequence(raw)
        features = build_features(positions, present)
        raw[0, C.LH, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "partially missing"):
            validate_recording_arrays(raw, positions, present, features)

    def test_export_requires_review_and_assignment_and_excludes_rejections(self):
        self._create_sample("train-1")
        self.store.review("train-1", "accepted")
        with self.assertRaisesRegex(ValueError, "verified ground truth"):
            self.store.assign_split("train-1", "train")
        self.store.verify_ground_truth(
            "train-1", "DOG1", "reviewer-a", "independently reviewed"
        )
        verified = self.store.get("train-1")
        self.assertEqual(verified["target_label"], "AXE1")
        self.assertEqual(verified["ground_truth_label"], "DOG1")
        self.assertEqual(verified["recording_status"], "VERIFIED")
        self.assertEqual(verified["verification_status"], "VERIFIED")
        self.assertEqual(verified["quality_status"], "PASSED")
        verified_sequence = load_attempt_sequence(
            Path(verified["sequence_path"])
        )
        self.assertEqual(verified_sequence["metadata"]["ground_truth_label"], "DOG1")
        self.assertEqual(
            verified_sequence["metadata"]["verification_status"], "VERIFIED"
        )
        self.store.assign_split("train-1", "train")
        with tempfile.TemporaryDirectory() as output_dir:
            archive_path = Path(output_dir) / "train.npz"
            summary = self.store.export_feature_archive(
                "train", archive_path, ["AXE1", "DOG1"]
            )
            self.assertEqual(summary["samples"], 1)
            with np.load(archive_path, allow_pickle=False) as archive:
                self.assertEqual(archive["y"].tolist(), [1])
                self.assertEqual(archive["source"].tolist(), ["self_recorded"])
                self.assertEqual(archive["sample_id"].tolist(), ["train-1"])
                self.assertEqual(
                    archive["ground_truth_label"].tolist(), ["DOG1"]
                )
                self.assertEqual(archive["target_label"].tolist(), ["AXE1"])
                self.assertEqual(archive["signer_id"].tolist(), ["S001"])
                self.assertEqual(
                    archive["dominant_hand"].tolist(), ["UNKNOWN"]
                )
                self.assertEqual(archive["camera_index"].tolist(), [0])

        self._create_sample("rejected-1", offset=0.2)
        self.store.review(
            "rejected-1", "rejected", rejection_reason="not the target sign"
        )
        with tempfile.TemporaryDirectory() as output_dir:
            count = self.store.export_split("train", Path(output_dir))
            with (Path(output_dir) / "manifest.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(count, 1)
            self.assertEqual([row["sample_id"] for row in rows], ["train-1"])
            self.assertTrue((Path(output_dir) / "train-1.npz").is_file())

    def test_signer_cannot_cross_splits_and_evaluation_split_is_locked(self):
        self._create_sample("train-1")
        self.store.review("train-1", "accepted")
        self.store.verify_ground_truth("train-1", "AXE1", "reviewer-01")
        self.store.assign_split("train-1", "train")

        self._create_sample("validation-1", signer_id="S001", offset=0.3)
        self.store.review("validation-1", "accepted")
        self.store.verify_ground_truth("validation-1", "AXE1", "reviewer-02")
        with self.assertRaisesRegex(ValueError, "signer overlap"):
            self.store.assign_split("validation-1", "validation")

        self.store.assign_split("train-1", "final_test")
        with self.assertRaisesRegex(ValueError, "locked"):
            self.store.assign_split("train-1", "train")

    def test_split_requires_anonymous_signer_id(self):
        with self.assertRaisesRegex(ValueError, "anonymous signer ID"):
            self._create_sample("anonymous-signer", signer_id="")
        with self.assertRaisesRegex(ValueError, "anonymous signer ID"):
            self._create_sample("named-signer", signer_id="Alice")
        self.assertEqual(
            SelfRecordedSampleStore.validate_signer_id("signer-01"),
            "signer-01",
        )

    def test_session_and_signer_metadata_are_required_and_persisted(self):
        self._create_sample("session-sample", session_id="session-123")
        self.assertEqual(self.store.get("session-sample")["session_id"], "session-123")

        with self.assertRaisesRegex(ValueError, "session ID"):
            self.store.create_pending(
                sample_id="missing-session",
                target_label="AXE1",
                signer_id="S001",
                camera_source="opencv-index:0",
                duration_seconds=1.0,
                original_frame_count=16,
                raw_landmarks=np.zeros((16, C.N_POINTS, 3), dtype=np.float32),
                normalized_positions=np.zeros(
                    (C.SEQ_LEN, C.N_POINTS, 3), dtype=np.float32
                ),
                hand_present=np.ones((C.SEQ_LEN, 2), dtype=np.float32),
                model_features=np.zeros((C.SEQ_LEN, 590), dtype=np.float32),
                model_version="checkpoint-sha",
            )

    def test_sessions_are_stable_across_samples_and_unique_between_sessions(self):
        first = self.store.new_session_id()
        second = self.store.new_session_id()
        self.assertNotEqual(first, second)
        self._create_sample("same-session-1", session_id=first)
        self._create_sample("same-session-2", session_id=first, offset=0.1)
        self._create_sample("other-session", session_id=second, offset=0.2)
        self.assertEqual(self.store.get("same-session-1")["session_id"], first)
        self.assertEqual(self.store.get("same-session-2")["session_id"], first)
        self.assertEqual(self.store.get("other-session")["session_id"], second)

    def test_legacy_accepted_record_is_unverified_and_missing_metadata_is_safe(self):
        legacy_path = self.root / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_path)
        try:
            connection.execute(
                """
                CREATE TABLE self_recorded_samples (
                    sample_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    target_label TEXT NOT NULL,
                    ground_truth_label TEXT,
                    signer_id TEXT NOT NULL DEFAULT '',
                    recorded_at TEXT NOT NULL,
                    sequence_path TEXT NOT NULL,
                    raw_video_path TEXT NOT NULL DEFAULT '',
                    dataset_split TEXT NOT NULL DEFAULT 'unassigned'
                )
                """
            )
            connection.execute(
                """
                INSERT INTO self_recorded_samples
                    (sample_id, status, target_label, ground_truth_label,
                     signer_id, recorded_at, sequence_path)
                VALUES ('legacy', 'accepted', 'AXE1', NULL, '', '2020-01-01T00:00:00+00:00', '')
                """
            )
            connection.commit()
        finally:
            connection.close()
        legacy_store = SelfRecordedSampleStore(legacy_path)
        legacy = legacy_store.get("legacy")
        self.assertEqual(legacy["recording_status"], "ACCEPTED_UNVERIFIED")
        self.assertIsNone(legacy["verification_status"])
        self.assertIsNone(legacy["quality_status"])
        self.assertIsNone(legacy["session_id"])
        self.assertEqual(legacy["ground_truth_label"], None)


if __name__ == "__main__":
    unittest.main()
