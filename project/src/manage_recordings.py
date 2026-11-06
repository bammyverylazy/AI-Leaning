"""Inspect, review, split, replay, and export self-recorded samples."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import config as C
from dataset import build_features, preprocess_sequence
from evaluation_sequences import load_attempt_sequence
from label_display import aggregate_predictions
from model import SeqClassifier
from model_vocabulary import checkpoint_vocabulary
from self_recorded_store import SelfRecordedSampleStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(C.SELF_RECORDING_DB))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dashboard", help="show collection coverage and counts")
    subparsers.add_parser("pending", help="list recordings awaiting review")
    inspect = subparsers.add_parser("inspect", help="inspect one saved sample")
    inspect.add_argument("sample_id")
    review = subparsers.add_parser("review", help="accept or reject a pending sample")
    review.add_argument("sample_id")
    review.add_argument("--decision", choices=("accepted", "rejected"), required=True)
    review.add_argument("--notes", default="")
    review.add_argument("--rejection-reason", default="")
    verify = subparsers.add_parser(
        "verify", help="record independently reviewed ground truth"
    )
    verify.add_argument("sample_id")
    verify.add_argument("--ground-truth", required=True)
    verify.add_argument("--reviewer-id", required=True)
    verify.add_argument("--review-notes", default="")
    split = subparsers.add_parser("assign-split", help="assign a reviewed sample to a split")
    split.add_argument("sample_id")
    split.add_argument(
        "--split",
        required=True,
        choices=("unassigned", "train", "validation", "test", "final_test", "real_world_eval"),
    )
    replay = subparsers.add_parser("replay", help="re-run inference from a saved sample")
    replay.add_argument("sample_id")
    replay.add_argument("--checkpoint", default=str(C.MODELS / "gru_seed3.pt"))
    export = subparsers.add_parser(
        "export", help="export reviewed accepted samples from one explicit split"
    )
    export.add_argument(
        "--split",
        required=True,
        choices=("train", "validation", "test", "final_test", "real_world_eval"),
    )
    export.add_argument("--output", required=True)
    prepare = subparsers.add_parser(
        "prepare-features",
        help="build separate train/val/test arrays from verified split assignments",
    )
    prepare.add_argument(
        "--output",
        default=str(C.SELF_RECORDING / "features"),
        help="output directory; archives are named train.npz, val.npz, and test.npz",
    )
    prepare.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing self-recorded feature archives",
    )
    args = parser.parse_args()

    database = Path(args.db)
    if not database.is_absolute():
        database = C.ROOT / database
    store = SelfRecordedSampleStore(database)
    if args.command == "dashboard":
        vocabulary = _load_vocab()
        rows = store.dashboard(vocabulary)
        print(
            f"Accepted: {rows['accepted']} · rejected: {rows['rejected']} · "
            f"signers: {rows['signers']} · classes covered: "
            f"{rows['classes_covered']}/{len(vocabulary)}"
        )
        print("Least-collected signs:")
        for item in rows["by_sign"][:20]:
            print(
                f"  {item['label']}: {item['samples']} samples, "
                f"{item['signers']} signers"
            )
        return
    if args.command == "pending":
        rows = store.samples(status="pending_review")
        if not rows:
            print("No recordings are awaiting review.")
            return
        for sample in rows:
            print(
                f"{sample['sample_id']} · target={sample['target_label']} · "
                f"signer={sample['signer_id'] or 'unknown'} · "
                f"session={sample['session_id'] or 'unknown'} · "
                f"recorded={sample['recorded_at']} · "
                f"sequence={sample['sequence_path']}"
            )
        return
    if args.command == "inspect":
        sample = store.get(args.sample_id)
        sequence = load_attempt_sequence(
            store.resolve_artifact_path(sample["sequence_path"])
        )
        print(f"Sample: {sample['sample_id']} ({sample['status']})")
        for name in (
            "target_label",
            "ground_truth_label",
            "reviewer_id",
            "signer_id",
            "session_id",
            "dominant_hand",
            "camera_source",
            "recording_status",
            "verification_status",
            "quality_status",
            "rejection_reason",
            "camera_index",
            "camera_angle",
            "distance",
            "lighting",
            "signing_speed",
            "notes",
            "dataset_split",
            "recorded_at",
            "duration_seconds",
            "original_frame_count",
            "hand_frame_count",
            "hand_frame_ratio",
            "pose_frame_count",
            "pose_frame_ratio",
            "shoulder_frame_count",
            "shoulder_frame_ratio",
            "preprocessing_version",
            "model_version",
            "sequence_path",
            "raw_video_path",
        ):
            print(f"{name}: {sample[name] or 'not recorded'}")
        print(
            "Array shapes: "
            + json.dumps({key: value.shape for key, value in sequence.items()
                          if isinstance(value, np.ndarray)})
        )
        print(f"Quality warnings: {sample['quality_warnings_json']}")
        return
    if args.command == "review":
        sample = store.review(
            args.sample_id,
            args.decision,
            args.notes,
            args.rejection_reason,
        )
        print(
            f"Sample {args.sample_id} marked {args.decision}; "
            f"recording status={sample['recording_status']}."
        )
        return
    if args.command == "verify":
        store.verify_ground_truth(
            args.sample_id,
            args.ground_truth,
            args.reviewer_id,
            args.review_notes,
        )
        print(f"Ground truth saved for sample {args.sample_id}.")
        return
    if args.command == "assign-split":
        store.assign_split(args.sample_id, args.split)
        print(f"Sample {args.sample_id} assigned to {args.split}.")
        return
    if args.command == "replay":
        sample = store.get(args.sample_id)
        sequence = load_attempt_sequence(
            store.resolve_artifact_path(sample["sequence_path"])
        )
        positions, present = preprocess_sequence(sequence["raw_landmarks"])
        recomputed = build_features(positions, present)
        preprocessing_matches = np.array_equal(
            recomputed, sequence["model_features"], equal_nan=True
        )
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = C.ROOT / checkpoint_path
        if not checkpoint_path.is_file():
            parser.error(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        vocab = checkpoint_vocabulary(checkpoint)
        if int(checkpoint["feature_dim"]) != sequence["model_features"].shape[1]:
            parser.error("Checkpoint feature dimension differs from saved sample.")
        model_args = checkpoint["args"]
        model = SeqClassifier(
            checkpoint["feature_dim"],
            len(vocab),
            model_args["hidden"],
            2,
            model_args["dropout"],
            model_args["kind"],
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()
        features = torch.from_numpy(
            np.asarray(sequence["model_features"], dtype=np.float32)
        )[None]
        with torch.no_grad():
            probabilities = torch.softmax(model(features), dim=1)[0]
        predictions = aggregate_predictions(
            probabilities, vocab, top_k=5
        )
        print(f"Sample: {args.sample_id}")
        print(f"Current checkpoint SHA-256: {checkpoint_hash}")
        print(
            "Checkpoint matches recorded model: "
            f"{checkpoint_hash == sample['model_version']}"
        )
        print(f"Preprocessing reproduces saved input: {preprocessing_matches}")
        for rank, prediction in enumerate(predictions, 1):
            print(f"{rank}. {prediction.canonical_label} ({prediction.confidence:.1%})")
        return
    if args.command == "export":
        output = Path(args.output)
        if not output.is_absolute():
            output = C.ROOT / output
        count = store.export_split(args.split, output)
        print(f"Exported {count} reviewed accepted samples from {args.split} to {output}")
        return
    if args.command == "prepare-features":
        output = Path(args.output)
        if not output.is_absolute():
            output = C.ROOT / output
        archive_paths = {
            "train": output / "train.npz",
            "validation": output / "val.npz",
            "test": output / "test.npz",
        }
        existing = [path for path in archive_paths.values() if path.exists()]
        if existing and not args.overwrite:
            parser.error(
                "Feature archive already exists; choose another --output or use "
                "--overwrite: " + ", ".join(str(path) for path in existing)
            )
        vocabulary = _load_vocab()
        for split, path in archive_paths.items():
            result = store.export_feature_archive(split, path, vocabulary)
            print(
                f"{result['split']}: {result['samples']} verified samples, "
                f"class counts={result['class_counts']} -> {result['output_path']}"
            )


def _load_vocab() -> list[str]:
    checkpoint_path = C.MODELS / "gru_seed3.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. "
            "Pass an available checkpoint before using vocabulary-dependent commands."
        )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    return checkpoint_vocabulary(checkpoint)


if __name__ == "__main__":
    main()
