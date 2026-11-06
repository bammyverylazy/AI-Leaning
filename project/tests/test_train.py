import sys
import tempfile
import unittest
import json
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C
from train import (
    append_self_recorded_split,
    build_class_mapping,
    classification_summary,
    ensure_candidate_checkpoint_path,
    load_self_recorded_feature_archive,
    validate_unique_self_recorded_splits,
)


class SelfRecordedTrainingIntegrationTests(unittest.TestCase):
    vocabulary = ["AXE1", "DOG1"]

    def _write_archive(
        self,
        path: Path,
        *,
        sample_id: str = "sample-1",
        signer_id: str = "S001",
        split: str = "train",
        recording_status: str = "VERIFIED",
        ground_truth: str = "DOG1",
        sequence_hash: str = hashlib.sha256(b"sequence-1").hexdigest(),
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            pos=np.zeros((1, C.SEQ_LEN, C.N_POINTS, 3), dtype=np.float32),
            present=np.zeros((1, C.SEQ_LEN, 2), dtype=np.float32),
            y=np.asarray([self.vocabulary.index(ground_truth)], dtype=np.int64),
            participant=np.asarray([f"self_recorded:{signer_id}"]),
            video=np.asarray([f"self_recorded:{sample_id}"]),
            source=np.asarray(["self_recorded"]),
            sample_id=np.asarray([sample_id]),
            signer_id=np.asarray([signer_id]),
            session_id=np.asarray(["session-1"]),
            target_label=np.asarray(["AXE1"]),
            ground_truth_label=np.asarray([ground_truth]),
            recorded_at=np.asarray(["2025-01-01T10:00:00+00:00"]),
            dominant_hand=np.asarray(["RIGHT"]),
            camera_source=np.asarray(["opencv-index:0"]),
            camera_index=np.asarray([0], dtype=np.int64),
            camera_angle=np.asarray(["front"]),
            distance=np.asarray(["normal"]),
            lighting=np.asarray(["normal"]),
            signing_speed=np.asarray(["normal"]),
            recording_status=np.asarray([recording_status]),
            verification_status=np.asarray(["VERIFIED"]),
            quality_status=np.asarray(["PASSED"]),
            dataset_split=np.asarray(["val" if split == "validation" else split]),
            sequence_sha256=np.asarray([sequence_hash]),
            vocab=np.asarray(self.vocabulary),
            class_mapping_json=np.asarray(
                json.dumps(build_class_mapping(self.vocabulary))
            ),
        )

    def test_only_verified_quality_passed_ground_truth_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            self._write_archive(path)
            data = load_self_recorded_feature_archive(
                path, "train", self.vocabulary
            )
        self.assertEqual(data["target_label"].tolist(), ["AXE1"])
        self.assertEqual(data["ground_truth_label"].tolist(), ["DOG1"])
        self.assertEqual(data["y"].tolist(), [1])
        self.assertEqual(data["source"].tolist(), ["self_recorded"])

    def test_unverified_or_wrong_split_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            self._write_archive(path, recording_status="ACCEPTED_UNVERIFIED")
            with self.assertRaisesRegex(ValueError, "recording_status"):
                load_self_recorded_feature_archive(
                    path, "train", self.vocabulary
                )
            self._write_archive(path, split="test")
            with self.assertRaisesRegex(ValueError, "dataset_split"):
                load_self_recorded_feature_archive(
                    path, "train", self.vocabulary
                )

    def test_same_sample_or_sequence_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.npz"
            test_path = Path(directory) / "test.npz"
            self._write_archive(train_path, split="train")
            self._write_archive(
                test_path,
                split="test",
                sequence_hash=hashlib.sha256(b"sequence-2").hexdigest(),
            )
            train_data = load_self_recorded_feature_archive(
                train_path, "train", self.vocabulary
            )
            test_data = load_self_recorded_feature_archive(
                test_path, "test", self.vocabulary
            )
        with self.assertRaisesRegex(ValueError, "duplicate a sample"):
            validate_unique_self_recorded_splits(
                {"train": train_data, "test": test_data}
            )

    def test_archive_is_added_with_signer_and_source_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.npz"
            self._write_archive(path)
            data = load_self_recorded_feature_archive(
                path, "train", self.vocabulary
            )
        dataset = type(
            "Dataset",
            (),
            {
                "pos": np.zeros((1, C.SEQ_LEN, C.N_POINTS, 3), dtype=np.float32),
                "present": np.zeros((1, C.SEQ_LEN, 2), dtype=np.float32),
                "y": np.asarray([0], dtype=np.int64),
                "participant": np.asarray(["source:signer"]),
                "video": np.asarray(["source-video"]),
            },
        )()
        summary = {
            "samples": 1,
            "counts": Counter({0: 1}),
            "participants": {"source:signer"},
        }
        append_self_recorded_split(dataset, summary, data)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(dataset.video[-1], "self_recorded:sample-1")
        self.assertEqual(dataset.participant[-1], "self_recorded:S001")
        self.assertEqual(dataset.y.tolist(), [0, 1])

    def test_evaluation_summary_has_class_metrics_and_confusion_matrix(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0]])
        summary = classification_summary(
            logits, torch.tensor([0, 1, 0]), self.vocabulary
        )
        self.assertEqual(summary["top1"], 2 / 3)
        self.assertEqual(summary["class_counts"][0]["samples"], 2)
        self.assertEqual(summary["confusion_matrix"], [[1, 1], [0, 1]])
        self.assertIn("precision", summary["per_class"][0])

    def test_baseline_checkpoint_cannot_be_a_candidate_output(self):
        with self.assertRaisesRegex(ValueError, "baseline checkpoint.*protected"):
            ensure_candidate_checkpoint_path(C.MODELS / "gru_seed3.pt")


if __name__ == "__main__":
    unittest.main()
