import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from audit_dataset import _self_recorded_data_report, audit_dataset
from dataset import FEATURE_DIM, build_features, preprocess_sequence
from model import SeqClassifier
from model_vocabulary import build_class_mapping
from self_recorded_store import SelfRecordedSampleStore
from train import validate_feature_archive


class DatasetCoverageTests(unittest.TestCase):
    def test_feature_validator_catches_misaligned_training_indices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            pd.DataFrame(
                [
                    {"video": "axe.mp4", "gloss": "AXE1"},
                    {"video": "dog.mp4", "gloss": "DOG1"},
                ]
            ).to_csv(data_dir / "subset_train.csv", index=False)
            archive_path = data_dir / "train.npz"
            np.savez_compressed(
                archive_path,
                pos=np.zeros((2, C.SEQ_LEN, C.N_POINTS, 3), dtype=np.float32),
                present=np.zeros((2, C.SEQ_LEN, 2), dtype=np.float32),
                y=np.asarray([0, 0], dtype=np.int64),
                video=np.asarray(["axe.mp4", "dog.mp4"]),
                participant=np.asarray(["signer-1", "signer-2"]),
            )

            with patch.object(C, "DATA", data_dir):
                with self.assertRaisesRegex(ValueError, "class indices do not match"):
                    validate_feature_archive(
                        "train", archive_path, ["AXE1", "DOG1"]
                    )

    def test_full_audit_reports_raw_only_and_checkpoint_supported_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            raw_dir = data_dir / "raw"
            split_dir = raw_dir / "splits"
            videos_dir = raw_dir / "videos"
            landmarks_dir = data_dir / "landmarks"
            features_dir = data_dir / "features"
            self_recorded_dir = data_dir / "self_recorded"
            models_dir = root / "models"
            for directory in (
                split_dir,
                videos_dir,
                landmarks_dir,
                features_dir,
                self_recorded_dir,
                models_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)

            active_records = []
            for split, signer in (("train", "s1"), ("val", "s2"), ("test", "s3")):
                video = f"{split}.mp4"
                pd.DataFrame(
                    [
                        {
                            "Participant ID": signer,
                            "Video File": video,
                            "Gloss": "AXE1",
                        },
                        {
                            "Participant ID": "raw-only",
                            "Video File": "unused.mp4",
                            "Gloss": "RAWONLY",
                        },
                    ]
                ).to_csv(split_dir / f"{split}.csv", index=False)
                pd.DataFrame(
                    [
                        {"participant": signer, "video": video, "gloss": "AXE1"}
                    ]
                ).to_csv(data_dir / f"subset_{split}.csv", index=False)
                (videos_dir / video).write_bytes(b"fixture video")
                raw = np.full((8, C.N_POINTS, 3), np.nan, dtype=np.float32)
                raw[:, C.LH, 0:2] = 0.4
                np.save(landmarks_dir / f"{Path(video).stem}.npy", raw)
                positions, present = preprocess_sequence(raw)
                np.savez_compressed(
                    features_dir / f"{split}.npz",
                    pos=positions[None, ...],
                    present=present[None, ...],
                    y=np.asarray([0], dtype=np.int64),
                    video=np.asarray([video]),
                    participant=np.asarray([signer]),
                )
                active_records.append({"split": split, "video": video})

            (data_dir / "vocab.txt").write_text("AXE1\n", encoding="utf-8")
            checkpoint_path = models_dir / "model.pt"
            model = SeqClassifier(FEATURE_DIM, 1, hidden=4, dropout=0.1, kind="gru")
            torch.save(
                {
                    "vocab": ["AXE1"],
                    "class_mapping": build_class_mapping(["AXE1"]),
                    "feature_dim": FEATURE_DIM,
                    "args": {"kind": "gru", "hidden": 4, "dropout": 0.1},
                    "model": model.state_dict(),
                },
                checkpoint_path,
            )

            with (
                patch.object(C, "ROOT", root),
                patch.object(C, "DATA", data_dir),
                patch.object(C, "RAW", raw_dir),
                patch.object(C, "LANDMARKS", landmarks_dir),
                patch.object(C, "FEATURES", features_dir),
                patch.object(C, "MODELS", models_dir),
                patch.object(C, "SELF_RECORDING", self_recorded_dir),
            ):
                report = audit_dataset(checkpoint_path)

            self.assertEqual(report["raw_dataset"]["unique_classes"], 2)
            self.assertEqual(report["active_vocabulary"]["class_count"], 1)
            self.assertEqual(report["selected_checkpoint"]["class_count"], 1)
            self.assertTrue(
                report["selected_checkpoint"]["explicit_class_mapping_present"]
            )
            self.assertTrue(report["selected_checkpoint"]["evaluation"]["available"])
            self.assertEqual(
                report["selected_checkpoint"]["evaluation"]["test_samples"], 1
            )
            self.assertEqual(report["ui_vocabulary"]["testing_mode_class_names"], ["AXE1"])
            self.assertFalse(
                report["training_data_safety"][
                    "self_recorded_samples_in_feature_generation"
                ]
            )
            self.assertFalse(report["self_recorded_data"]["available"])
            for replay in report["preprocessing_replay"].values():
                self.assertEqual(replay["replayed_samples"], 1)
                self.assertEqual(replay["mismatched_samples"], 0)
            self.assertIn("RAWONLY", report["class_lists"]["raw_missing_from_preprocessing"])
            raw_only = next(
                row for row in report["class_coverage"] if row["label"] == "RAWONLY"
            )
            supported = next(
                row for row in report["class_coverage"] if row["label"] == "AXE1"
            )
            self.assertIn("RAW_ONLY", raw_only["statuses"])
            self.assertIn("SUPPORTED_BY_MODEL", supported["statuses"])

    def test_self_recorded_report_checks_replay_and_missing_signer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recording_dir = Path(temp_dir) / "self_recorded"
            with (
                patch.object(C, "SELF_RECORDING", recording_dir),
                patch.object(C, "SELF_RECORDING_DB", recording_dir / "samples.sqlite3"),
            ):
                store = SelfRecordedSampleStore()
                raw = np.full((16, C.N_POINTS, 3), np.nan, dtype=np.float32)
                raw[:, C.LH, :] = (0.4, 0.5, 0.0)
                positions, present = preprocess_sequence(raw)
                store.create_pending(
                    sample_id="pending-sample",
                    target_label="AXE1",
                    signer_id="S001",
                    session_id="session-1",
                    dominant_hand="UNKNOWN",
                    camera_source="opencv-index:0",
                    duration_seconds=1.0,
                    original_frame_count=len(raw),
                    raw_landmarks=raw,
                    normalized_positions=positions,
                    hand_present=present,
                    model_features=build_features(positions, present),
                    model_version="test-checkpoint",
                )
                with store._connect() as connection:
                    connection.execute(
                        """
                        UPDATE self_recorded_samples
                        SET signer_id = '', session_id = NULL
                        WHERE sample_id = 'pending-sample'
                        """
                    )
                    legacy_row = connection.execute(
                        "SELECT * FROM self_recorded_samples "
                        "WHERE sample_id = 'pending-sample'"
                    ).fetchone()
                    store._write_sequence_metadata(
                        Path(legacy_row["sequence_path"]), legacy_row
                    )

                pending_report = _self_recorded_data_report()
                store.review("pending-sample", "accepted")
                accepted_report = _self_recorded_data_report()

            self.assertEqual(pending_report["sample_count"], 1)
            self.assertEqual(pending_report["status_counts"]["pending_review"], 1)
            self.assertEqual(pending_report["unique_target_labels"], 1)
            self.assertEqual(pending_report["unique_nonempty_signers"], 0)
            self.assertEqual(pending_report["missing_metadata_counts"]["signer_id"], 1)
            self.assertEqual(pending_report["missing_metadata_counts"]["session_id"], 1)
            self.assertEqual(pending_report["replayable_sample_count"], 1)
            self.assertEqual(pending_report["invalid_sequence_count"], 0)
            self.assertFalse(pending_report["training_inclusion"])
            self.assertEqual(accepted_report["status_counts"]["accepted"], 1)
            self.assertEqual(accepted_report["ground_truth_verified_count"], 0)
            self.assertEqual(accepted_report["accepted_without_ground_truth_count"], 1)


if __name__ == "__main__":
    unittest.main()
