"""Audit raw ASL data, preprocessing, feature splits, checkpoints, and UI coverage."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import config as C
from dataset import FEATURE_DIM, build_features, preprocess_sequence
from label_display import canonical_label
from model import SeqClassifier
from model_vocabulary import checkpoint_vocabulary, validate_vocabulary
from evaluation_sequences import load_attempt_sequence
from self_recorded_store import validate_recording_arrays


_SPLITS = ("train", "val", "test")
_MIN_TRAIN_SAMPLES = 15
_MIN_VALIDATION_SAMPLES = 3
_MIN_TEST_SAMPLES = 8
_MIN_HAND_RATIO = 0.2


def _column(frame: pd.DataFrame, *aliases: str) -> str:
    columns = {
        str(name).strip().lower().replace("_", " "): str(name)
        for name in frame.columns
    }
    for alias in aliases:
        found = columns.get(alias.lower().replace("_", " "))
        if found is not None:
            return found
    raise ValueError(
        f"Could not find any of the columns {aliases} in {list(frame.columns)}."
    )


def _load_raw_split(split: str) -> pd.DataFrame:
    source = pd.read_csv(C.find_split_csv(split))
    return pd.DataFrame(
        {
            "video": source[_column(source, "video file", "video", "filename", "file")]
            .astype(str)
            .map(lambda value: Path(str(value)).name),
            "label": source[_column(source, "gloss", "sign", "label")].astype(str),
            "signer": source[
                _column(source, "participant id", "participant", "signer", "user")
            ].astype(str),
        }
    )


def _feature_summary(
    split: str,
    vocab: list[str],
    subset_frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], set[str], list[dict[str, Any]]]:
    path = C.FEATURES / f"{split}.npz"
    subset = subset_frames[split]
    label_by_video = dict(zip(subset["video"], subset["label"]))
    summary: dict[str, Any] = {
        "path": str(path.relative_to(C.ROOT)),
        "exists": path.is_file(),
        "samples": 0,
        "classes": [],
        "class_counts": {},
        "sample_class_mapping_matches_subset_and_vocab": None,
        "shape_errors": [],
    }
    if not path.is_file():
        return summary, set(), []

    with np.load(path, allow_pickle=True) as archive:
        required = {"pos", "present", "y", "video", "participant"}
        missing = sorted(required - set(archive.files))
        summary["missing_arrays"] = missing
        if missing:
            return summary, set(), []
        positions = archive["pos"]
        present = archive["present"]
        indices = np.asarray(archive["y"], dtype=np.int64)
        videos = np.asarray(archive["video"]).astype(str)
        participants = np.asarray(archive["participant"]).astype(str)
        summary["samples"] = int(len(indices))
        summary["arrays"] = {
            name: list(archive[name].shape)
            for name in ("pos", "present", "y", "participant", "video")
        }
        expected_shapes = {
            "pos": (len(indices), C.SEQ_LEN, C.N_POINTS, 3),
            "present": (len(indices), C.SEQ_LEN, 2),
            "y": (len(indices),),
            "participant": (len(indices),),
            "video": (len(indices),),
        }
        actual_shapes = {
            "pos": positions.shape,
            "present": present.shape,
            "y": indices.shape,
            "participant": participants.shape,
            "video": videos.shape,
        }
        summary["shape_errors"] = [
            f"{name}: expected {expected_shapes[name]}, got {actual_shapes[name]}"
            for name in expected_shapes
            if expected_shapes[name] != actual_shapes[name]
        ]
        counts = Counter(
            vocab[index] if 0 <= index < len(vocab) else f"INVALID_INDEX_{index}"
            for index in map(int, indices)
        )
        summary["classes"] = sorted(label for label in counts if label in vocab)
        summary["class_counts"] = dict(sorted(counts.items()))
        summary["invalid_indices"] = sorted(
            {int(index) for index in indices if index < 0 or index >= len(vocab)}
        )
        unknown_videos = sorted(set(videos) - set(label_by_video))
        mapping_matches = not unknown_videos and not summary["invalid_indices"]
        index_by_label = {label: index for index, label in enumerate(vocab)}
        for video, index in zip(videos, indices):
            expected_index = index_by_label.get(label_by_video.get(video, ""))
            if expected_index != int(index):
                mapping_matches = False
        summary["sample_class_mapping_matches_subset_and_vocab"] = mapping_matches
        summary["unknown_videos"] = unknown_videos
        if "vocab" in archive.files:
            archive_vocab = [str(label) for label in archive["vocab"].tolist()]
            summary["embedded_vocabulary"] = archive_vocab
            summary["embedded_vocabulary_matches_active"] = archive_vocab == vocab
        else:
            summary["embedded_vocabulary"] = None
            summary["embedded_vocabulary_matches_active"] = None
        if "class_mapping_json" in archive.files:
            try:
                mapping = json.loads(str(archive["class_mapping_json"].item()))
                summary["embedded_class_mapping_matches_active"] = mapping == [
                    {
                        "class_index": i,
                        "label": label,
                        "canonical_label": canonical_label(label),
                    }
                    for i, label in enumerate(vocab)
                ]
            except (TypeError, ValueError, json.JSONDecodeError):
                summary["embedded_class_mapping_matches_active"] = False
        else:
            summary["embedded_class_mapping_matches_active"] = None
    return summary, set(summary["classes"]), []


def _replay_feature_archive(split: str) -> dict[str, Any]:
    archive_path = C.FEATURES / f"{split}.npz"
    if not archive_path.is_file():
        return {
            "available": False,
            "source_samples": 0,
            "replayed_samples": 0,
            "missing_landmarks": 0,
            "mismatched_samples": 0,
            "invalid_sequences": 0,
        }
    replayed = 0
    missing_landmarks = 0
    mismatched = 0
    invalid = 0
    with np.load(archive_path, allow_pickle=True) as archive:
        videos = np.asarray(archive["video"]).astype(str)
        positions = archive["pos"]
        present = archive["present"]
        for index, video in enumerate(videos):
            landmark_path = C.LANDMARKS / f"{Path(video).stem}.npy"
            if not landmark_path.is_file():
                missing_landmarks += 1
                continue
            try:
                raw = np.load(landmark_path, mmap_mode="r")
                replayed_positions, replayed_present = preprocess_sequence(raw)
                replayed_features = build_features(
                    replayed_positions, replayed_present
                )
                if (
                    not np.isfinite(replayed_features).all()
                    or not np.array_equal(replayed_positions, positions[index])
                    or not np.array_equal(replayed_present, present[index])
                    or replayed_features.shape != (C.SEQ_LEN, FEATURE_DIM)
                ):
                    mismatched += 1
                    continue
                replayed += 1
            except (OSError, ValueError, EOFError, IndexError):
                invalid += 1
    return {
        "available": True,
        "source_samples": len(videos),
        "replayed_samples": replayed,
        "missing_landmarks": missing_landmarks,
        "mismatched_samples": mismatched,
        "invalid_sequences": invalid,
        "pipeline": "preprocess_sequence -> build_features",
    }


def _self_recorded_data_report() -> dict[str, Any]:
    def text_value(value: Any) -> str:
        return "" if value is None else str(value).strip()

    root = C.SELF_RECORDING.resolve()
    database_path = root / "samples.sqlite3"
    if not database_path.is_file():
        return {
            "available": False,
            "reason": "Self-recording database does not exist.",
            "sample_count": 0,
            "status_counts": {
                "accepted": 0,
                "rejected": 0,
                "pending_review": 0,
            },
            "target_labels": [],
            "unique_target_labels": 0,
            "unique_nonempty_signers": 0,
            "ground_truth_verified_count": 0,
            "accepted_without_ground_truth_count": 0,
            "target_ground_truth_mismatch_count": 0,
            "missing_metadata_counts": {},
            "replayable_sample_count": 0,
            "missing_sequence_count": 0,
            "invalid_sequence_count": 0,
            "missing_referenced_video_count": 0,
            "orphan_sequence_file_count": 0,
            "training_inclusion": False,
        }

    connection = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute(
            "SELECT * FROM self_recorded_samples"
        )]
    finally:
        connection.close()

    status_counts = Counter(text_value(row.get("status")) for row in rows)
    target_labels = {
        text_value(row.get("target_label"))
        for row in rows
        if text_value(row.get("target_label"))
    }
    signers = {
        text_value(row.get("signer_id"))
        for row in rows
        if text_value(row.get("signer_id"))
    }
    required_metadata = (
        "target_label",
        "signer_id",
        "session_id",
        "dominant_hand",
        "camera_source",
        "camera_angle",
        "distance",
        "lighting",
        "signing_speed",
        "recorded_at",
        "preprocessing_version",
        "model_version",
    )
    missing_metadata: Counter[str] = Counter()
    replayable = 0
    missing_sequences = 0
    invalid_sequences = 0
    missing_videos = 0
    sequence_paths = set()
    for row in rows:
        for field in required_metadata:
            if not text_value(row.get(field)):
                missing_metadata[field] += 1
        sequence_value = text_value(row.get("sequence_path"))
        sequence_path = Path(sequence_value).resolve() if sequence_value else None
        if sequence_path is None:
            missing_sequences += 1
        else:
            try:
                sequence_path.relative_to(root)
            except ValueError:
                invalid_sequences += 1
            else:
                sequence_paths.add(sequence_path)
                if not sequence_path.is_file():
                    missing_sequences += 1
                else:
                    try:
                        sequence = load_attempt_sequence(sequence_path)
                        validate_recording_arrays(
                            sequence["raw_landmarks"],
                            sequence["normalized_positions"],
                            sequence["hand_present"],
                            sequence["model_features"],
                        )
                    except (OSError, ValueError, EOFError, KeyError, IndexError):
                        invalid_sequences += 1
                    else:
                        replayable += 1
        video_value = text_value(row.get("raw_video_path"))
        if video_value:
            video_path = Path(video_value).resolve()
            try:
                video_path.relative_to(root)
            except ValueError:
                missing_videos += 1
            else:
                if not video_path.is_file():
                    missing_videos += 1

    stored_sequence_files = {
        path.resolve()
        for path in root.rglob("*.npz")
        if path.is_file()
    }
    return {
        "available": True,
        "sample_count": len(rows),
        "status_counts": {
            status: int(status_counts.get(status, 0))
            for status in ("accepted", "rejected", "pending_review")
        },
        "target_labels": sorted(target_labels),
        "unique_target_labels": len(target_labels),
        "unique_nonempty_signers": len(signers),
        "ground_truth_verified_count": sum(
            bool(text_value(row.get("ground_truth_label")))
            and row.get("verification_status") == "VERIFIED"
            and row.get("recording_status") == "VERIFIED"
            for row in rows
        ),
        "accepted_without_ground_truth_count": sum(
            row.get("status") == "accepted"
            and not text_value(row.get("ground_truth_label"))
            for row in rows
        ),
        "target_ground_truth_mismatch_count": sum(
            bool(text_value(row.get("ground_truth_label")))
            and text_value(row.get("target_label"))
            != text_value(row.get("ground_truth_label"))
            for row in rows
        ),
        "missing_metadata_counts": dict(sorted(missing_metadata.items())),
        "replayable_sample_count": replayable,
        "missing_sequence_count": missing_sequences,
        "invalid_sequence_count": invalid_sequences,
        "missing_referenced_video_count": missing_videos,
        "orphan_sequence_file_count": len(stored_sequence_files - sequence_paths),
        "training_inclusion": False,
    }


def _evaluate_selected_checkpoint(
    checkpoint: dict[str, Any],
    vocabulary: list[str],
    test_archive: Path,
) -> dict[str, Any]:
    if not test_archive.is_file():
        return {"available": False, "reason": "Test feature archive is missing."}
    args = checkpoint.get("args")
    state = checkpoint.get("model")
    if (
        not isinstance(args, dict)
        or not {"kind", "hidden", "dropout"}.issubset(args)
        or not isinstance(state, dict)
    ):
        return {
            "available": False,
            "reason": "Checkpoint lacks model arguments or a complete model state.",
        }
    with np.load(test_archive, allow_pickle=True) as data:
        positions = data["pos"]
        present = data["present"]
        labels = np.asarray(data["y"], dtype=np.int64)
        if not len(labels):
            return {"available": False, "reason": "Test feature archive is empty."}
        features = np.stack(
            [
                build_features(positions[index], present[index])
                for index in range(len(labels))
            ]
        )
    if features.shape[2] != FEATURE_DIM or checkpoint.get("feature_dim") != FEATURE_DIM:
        return {
            "available": False,
            "reason": "Checkpoint and test feature dimensions do not match.",
        }
    if labels.min() < 0 or labels.max() >= len(vocabulary):
        return {
            "available": False,
            "reason": "Test archive contains a class index outside the checkpoint vocabulary.",
        }
    model = SeqClassifier(
        FEATURE_DIM,
        len(vocabulary),
        int(args["hidden"]),
        2,
        float(args["dropout"]),
        str(args["kind"]),
    )
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features))
        top_indices = logits.topk(min(5, len(vocabulary)), dim=1).indices.numpy()
    predictions = logits.argmax(dim=1).numpy()
    class_count = len(vocabulary)
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    per_class = []
    for index, label in enumerate(vocabulary):
        support = int((labels == index).sum())
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class.append(
            {
                "class_index": index,
                "label": label,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "available": True,
        "test_samples": int(len(labels)),
        "top1_correct": int((predictions == labels).sum()),
        "top1_accuracy": float((predictions == labels).mean()),
        "top5_correct": int(
            (top_indices == labels[:, None]).any(axis=1).sum()
        ),
        "top5_accuracy": float(
            (top_indices == labels[:, None]).any(axis=1).mean()
        ),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "evaluation_source": str(test_archive.relative_to(C.ROOT)),
        "note": (
            "Source-dataset evaluation only; this does not establish real-world "
            "webcam accuracy."
        ),
    }


def audit_dataset(checkpoint_path: Path) -> dict[str, Any]:
    active_vocab = validate_vocabulary(
        [
            label.strip()
            for label in (C.DATA / "vocab.txt").read_text(encoding="utf-8").splitlines()
            if label.strip()
        ]
    )
    raw_splits = {split: _load_raw_split(split) for split in _SPLITS}
    raw_labels = set().union(*(set(frame["label"]) for frame in raw_splits.values()))
    raw_rows = pd.concat(
        [frame.assign(split=split) for split, frame in raw_splits.items()],
        ignore_index=True,
    )
    raw_class_counts = {
        split: dict(sorted(Counter(frame["label"]).items()))
        for split, frame in raw_splits.items()
    }
    raw_classes_by_split = {
        split: sorted(set(frame["label"]))
        for split, frame in raw_splits.items()
    }

    local_video_dir = C.find_videos_dir()
    local_video_paths = sorted(local_video_dir.rglob("*.mp4"))
    local_video_names = {path.name for path in local_video_paths}
    metadata_rows_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows.to_dict("records"):
        metadata_rows_by_video[row["video"]].append(
            {"label": row["label"], "signer": row["signer"], "split": row["split"]}
        )
    local_metadata = {
        name: metadata_rows_by_video.get(name, [])
        for name in local_video_names
    }
    quality_path = C.LANDMARKS / "_quality.csv"
    quality = (
        pd.read_csv(quality_path).fillna("")
        if quality_path.is_file()
        else pd.DataFrame(columns=("video", "status", "frames", "hand_ratio"))
    )
    quality_by_name: dict[str, list[str]] = defaultdict(list)
    for row in quality.to_dict("records"):
        quality_by_name[Path(str(row.get("video", ""))).name].append(
            str(row.get("status", ""))
        )
    local_video_labels = {
        item["label"]
        for entries in local_metadata.values()
        for item in entries
    }
    local_video_statuses = Counter(
        status for statuses in quality_by_name.values() for status in statuses
    )
    local_extraction_ok_names = {
        name for name, statuses in quality_by_name.items() if "ok" in statuses
    }
    local_extraction_ok_labels = {
        item["label"]
        for name in local_extraction_ok_names
        for item in local_metadata.get(name, [])
    }

    subset_frames = {}
    subset_counts = {}
    subset_classes = {}
    for split in _SPLITS:
        path = C.DATA / f"subset_{split}.csv"
        frame = pd.read_csv(path)
        frame = frame.rename(
            columns={
                _column(frame, "video", "video file", "filename", "file"): "video",
                _column(frame, "gloss", "sign", "label"): "label",
            }
        )
        frame["video"] = frame["video"].astype(str).map(
            lambda value: Path(str(value)).name
        )
        frame["label"] = frame["label"].astype(str)
        subset_frames[split] = frame
        subset_counts[split] = dict(sorted(Counter(frame["label"]).items()))
        subset_classes[split] = sorted(set(frame["label"]))

    feature_summaries = {}
    feature_classes_by_split = {}
    feature_videos_by_split = {}
    preprocessing_replay = {}
    for split in _SPLITS:
        summary, classes, _ = _feature_summary(split, active_vocab, subset_frames)
        feature_summaries[split] = summary
        feature_classes_by_split[split] = classes
        preprocessing_replay[split] = _replay_feature_archive(split)
        feature_videos_by_split[split] = set()
        feature_path = C.FEATURES / f"{split}.npz"
        if feature_path.is_file():
            with np.load(feature_path, allow_pickle=True) as archive:
                if "video" in archive.files:
                    feature_videos_by_split[split] = set(
                        np.asarray(archive["video"]).astype(str)
                    )

    preprocessing_exclusions = []
    preprocessing_failure_counts: Counter[str] = Counter()
    excluded_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    for split in _SPLITS:
        for row in subset_frames[split].to_dict("records"):
            if row["video"] in feature_videos_by_split[split]:
                continue
            landmark_path = C.LANDMARKS / f"{Path(row['video']).stem}.npy"
            exclusion: dict[str, Any] = {
                "split": split,
                "video": row["video"],
                "label": row["label"],
                "reason": "",
                "details": "",
            }
            if not landmark_path.is_file():
                exclusion["reason"] = "missing_landmarks"
                exclusion["details"] = "Expected landmark .npy file is missing."
            else:
                try:
                    landmarks = np.load(landmark_path, mmap_mode="r")
                    if landmarks.ndim != 3 or landmarks.shape[1:] != (C.N_POINTS, 3):
                        exclusion["reason"] = "invalid_landmark_shape"
                        exclusion["details"] = str(landmarks.shape)
                    elif len(landmarks) == 0:
                        exclusion["reason"] = "empty_landmarks"
                        exclusion["details"] = "Landmark sequence has zero frames."
                    else:
                        hand_frames = (
                            ~np.isnan(landmarks[:, C.LH, 0]).any(axis=1)
                            | ~np.isnan(landmarks[:, C.RH, 0]).any(axis=1)
                        )
                        ratio = float(hand_frames.mean())
                        if ratio < _MIN_HAND_RATIO:
                            exclusion["reason"] = "low_hand_ratio"
                            exclusion["details"] = (
                                f"{ratio:.4f}; minimum is {_MIN_HAND_RATIO:.4f}."
                            )
                        else:
                            exclusion["reason"] = "not_in_feature_archive"
                            exclusion["details"] = (
                                "Landmarks pass the standard filter, but the video "
                                "is absent from the saved feature archive."
                            )
                except (OSError, ValueError, EOFError) as exc:
                    exclusion["reason"] = "unreadable_landmarks"
                    exclusion["details"] = str(exc)
            preprocessing_exclusions.append(exclusion)
            preprocessing_failure_counts[exclusion["reason"]] += 1
            excluded_by_label[row["label"]][exclusion["reason"]] += 1

    checkpoints = []
    for path in sorted(C.MODELS.glob("*.pt")):
        if path.suffix.lower() != ".pt":
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint_vocab = checkpoint_vocabulary(checkpoint)
            model_state = checkpoint.get("model", {})
            output_weight = model_state.get("head.1.weight")
            output_count = int(output_weight.shape[0]) if output_weight is not None else None
            checkpoints.append(
                {
                    "path": str(path.relative_to(C.ROOT)),
                    "classes": len(checkpoint_vocab),
                    "vocabulary": checkpoint_vocab,
                    "explicit_class_mapping_present": "class_mapping" in checkpoint,
                    "feature_dimension": checkpoint.get("feature_dim"),
                    "output_dimension": output_count,
                    "output_matches_vocabulary": output_count == len(checkpoint_vocab),
                    "matches_active_vocab": checkpoint_vocab == active_vocab,
                    "error": None,
                }
            )
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            checkpoints.append(
                {
                    "path": str(path.relative_to(C.ROOT)),
                    "classes": 0,
                    "vocabulary": [],
                    "explicit_class_mapping_present": False,
                    "feature_dimension": None,
                    "output_dimension": None,
                    "output_matches_vocabulary": False,
                    "matches_active_vocab": False,
                    "error": str(exc),
                }
            )

    selected_path = checkpoint_path
    if not selected_path.is_absolute():
        selected_path = C.ROOT / selected_path
    selected_checkpoint: dict[str, Any] | None = None
    selected_vocab: list[str] = []
    selected_error = None
    if selected_path.is_file():
        try:
            loaded_checkpoint = torch.load(
                selected_path, map_location="cpu", weights_only=False
            )
            if not isinstance(loaded_checkpoint, dict):
                raise ValueError("Selected checkpoint must contain a dictionary.")
            selected_checkpoint = loaded_checkpoint
            selected_vocab = checkpoint_vocabulary(loaded_checkpoint)
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            selected_error = str(exc)
    else:
        selected_error = f"Checkpoint not found: {selected_path}"
    selected_output_count = None
    if selected_checkpoint is not None:
        selected_state = selected_checkpoint.get("model", {})
        output_weight = selected_state.get("head.1.weight")
        if output_weight is not None:
            selected_output_count = int(output_weight.shape[0])

    features_union = set().union(*feature_classes_by_split.values())
    train_classes = feature_classes_by_split["train"]
    selected_set = set(selected_vocab)
    classes = sorted(raw_labels | set(active_vocab) | features_union | selected_set)
    class_rows = []
    for label in classes:
        subset_by_split = {
            split: int(subset_counts[split].get(label, 0)) for split in _SPLITS
        }
        feature_count_by_split = {
            split: int(feature_summaries[split]["class_counts"].get(label, 0))
            for split in _SPLITS
        }
        statuses = []
        if label in raw_labels and label not in set(active_vocab):
            statuses.append("RAW_ONLY")
        if excluded_by_label.get(label):
            statuses.append("PREPROCESSING_FAILED")
        if label in features_union and label not in selected_set:
            statuses.append("FEATURES_ONLY")
        if label in features_union and not feature_count_by_split["train"]:
            statuses.append("NOT_IN_TRAINING")
        if label in set(active_vocab) and label not in selected_set:
            statuses.append("NOT_IN_CHECKPOINT")
        if label in selected_set and feature_count_by_split["train"]:
            statuses.append("SUPPORTED_BY_MODEL")
        if label in set(active_vocab) and (
            feature_count_by_split["train"] < _MIN_TRAIN_SAMPLES
            or feature_count_by_split["val"] < _MIN_VALIDATION_SAMPLES
            or feature_count_by_split["test"] < _MIN_TEST_SAMPLES
        ):
            statuses.append("LOW_SAMPLE_COUNT")
        class_rows.append(
            {
                "label": label,
                "in_raw_metadata": label in raw_labels,
                "in_local_raw_videos": label in local_video_labels,
                "in_active_subset": label in set(active_vocab),
                "in_features": label in features_union,
                "in_training": label in train_classes,
                "in_selected_checkpoint": label in selected_set,
                "subset_video_counts": subset_by_split,
                "feature_sample_counts": feature_count_by_split,
                "preprocessing_exclusions": dict(
                    sorted(excluded_by_label.get(label, {}).items())
                ),
                "statuses": statuses,
            }
        )

    if selected_checkpoint is not None and selected_vocab == active_vocab:
        checkpoint_evaluation = _evaluate_selected_checkpoint(
            selected_checkpoint, selected_vocab, C.FEATURES / "test.npz"
        )
    else:
        checkpoint_evaluation = {
            "available": False,
            "reason": "Selected checkpoint does not match the active vocabulary.",
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(C.ROOT),
        "thresholds": {
            "minimum_train_samples": _MIN_TRAIN_SAMPLES,
            "minimum_validation_samples": _MIN_VALIDATION_SAMPLES,
            "minimum_test_samples": _MIN_TEST_SAMPLES,
            "minimum_hand_ratio": _MIN_HAND_RATIO,
        },
        "raw_dataset": {
            "metadata_video_rows_by_split": {
                split: len(frame) for split, frame in raw_splits.items()
            },
            "metadata_video_rows_total": len(raw_rows),
            "unique_metadata_video_names": int(raw_rows["video"].nunique()),
            "unique_classes": len(raw_labels),
            "class_names": sorted(raw_labels),
            "class_names_by_split": raw_classes_by_split,
            "class_video_counts_by_split": raw_class_counts,
            "local_mp4_count": len(local_video_paths),
            "local_videos_matched_to_metadata": sum(
                bool(entries) for entries in local_metadata.values()
            ),
            "local_video_class_names": sorted(local_video_labels),
            "local_video_class_count": len(local_video_labels),
            "local_extraction_ok_video_count": len(local_extraction_ok_names),
            "classes_with_extraction_ok_videos": sorted(local_extraction_ok_labels),
            "classes_with_extraction_ok_video_count": len(local_extraction_ok_labels),
            "previous_extraction_status_counts": dict(sorted(local_video_statuses.items())),
            "extraction_quality_report": (
                str(quality_path.relative_to(C.ROOT)) if quality_path.is_file() else None
            ),
        },
        "active_vocabulary": {
            "path": str((C.DATA / "vocab.txt").relative_to(C.ROOT)),
            "class_names": active_vocab,
            "class_count": len(active_vocab),
            "unique_labels": len(set(active_vocab)) == len(active_vocab),
        },
        "active_subset": {
            split: {
                "sample_count": len(subset_frames[split]),
                "class_count": len(subset_classes[split]),
                "class_names": subset_classes[split],
                "class_video_counts": subset_counts[split],
            }
            for split in _SPLITS
        },
        "preprocessing": {
            "classes_with_features_by_split": {
                split: sorted(feature_classes_by_split[split]) for split in _SPLITS
            },
            "classes_preprocessed": sorted(features_union),
            "class_count_preprocessed": len(features_union),
            "excluded_clip_count": len(preprocessing_exclusions),
            "exclusion_counts": dict(sorted(preprocessing_failure_counts.items())),
            "excluded_clips": preprocessing_exclusions,
        },
        "features": feature_summaries,
        "preprocessing_replay": preprocessing_replay,
        "checkpoints": checkpoints,
        "selected_checkpoint": {
            "path": str(selected_path),
            "error": selected_error,
            "class_count": len(selected_vocab),
            "class_names": selected_vocab,
            "output_count": selected_output_count,
            "explicit_class_mapping_present": (
                "class_mapping" in selected_checkpoint
                if selected_checkpoint is not None
                else False
            ),
            "evaluation": checkpoint_evaluation,
            "training_class_names_missing_from_checkpoint": sorted(
                set(active_vocab) - selected_set
            ),
            "checkpoint_class_names_missing_from_training": sorted(
                selected_set - set(active_vocab)
            ),
        },
        "ui_vocabulary": {
            "testing_mode_class_names": selected_vocab,
            "demo_mode_prediction_class_names": selected_vocab,
            "testing_equals_checkpoint": True,
            "demo_equals_testing": True,
            "note": (
                "Both application modes derive recognition labels from the loaded "
                "checkpoint; the curated demo catalog only controls reference videos."
            ),
        },
        "class_coverage": class_rows,
        "class_lists": {
            "raw_metadata": sorted(raw_labels),
            "active_subset": active_vocab,
            "preprocessed": sorted(features_union),
            "training": sorted(train_classes),
            "selected_checkpoint": selected_vocab,
            "testing_mode": selected_vocab,
            "demo_mode": selected_vocab,
            "raw_missing_from_preprocessing": sorted(raw_labels - features_union),
            "preprocessed_missing_from_training": sorted(features_union - train_classes),
            "training_missing_from_checkpoint": sorted(set(active_vocab) - selected_set),
            "checkpoint_missing_from_training": sorted(selected_set - set(active_vocab)),
        },
        "self_recorded_data": _self_recorded_data_report(),
        "signer_independence": _signer_split_report(subset_frames),
        "training_data_safety": {
            "self_recorded_samples_in_feature_generation": False,
            "basis": (
                "dataset.py builds train/val/test arrays only from data/subset_*.csv "
                "and data/landmarks; it does not read data/self_recorded."
            ),
            "official_test_split_exists": True,
            "dedicated_untouched_final_test_split_exists": False,
            "caution": (
                "The official test split is signer-separated but has existing model "
                "evaluation reports; reserve newly collected, independently assigned "
                "signers for any future final real-world evaluation."
            ),
        },
    }


def _signer_split_report(subset_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    signer_column = None
    # Subset CSVs use the standardized participant field generated by make_vocab.py.
    for frame in subset_frames.values():
        if "participant" in frame.columns:
            signer_column = "participant"
            break
    if signer_column is None:
        return {"available": False, "overlap": {}}
    signer_sets = {
        split: set(frame[signer_column].astype(str))
        for split, frame in subset_frames.items()
    }
    overlap = {
        f"{left}/{right}": sorted(signer_sets[left] & signer_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    return {
        "available": True,
        "signers_by_split": {
            split: len(values) for split, values in signer_sets.items()
        },
        "overlap": overlap,
    }


def _render_text(report: dict[str, Any]) -> str:
    lines = [
        "ASL DATASET AND MODEL COVERAGE AUDIT",
        f"Generated: {report['generated_at']}",
        "",
        "[Raw metadata and local video files]",
        f"Metadata clips: {report['raw_dataset']['metadata_video_rows_total']}",
        f"Metadata classes: {report['raw_dataset']['unique_classes']}",
        f"Local MP4 files: {report['raw_dataset']['local_mp4_count']}",
        f"Local videos matched to metadata: "
        f"{report['raw_dataset']['local_videos_matched_to_metadata']}",
        f"Previously extraction-OK videos/classes: "
        f"{report['raw_dataset']['local_extraction_ok_video_count']}/"
        f"{report['raw_dataset']['classes_with_extraction_ok_video_count']}",
        "Classes: " + ", ".join(report["raw_dataset"]["class_names"]),
        "",
        "[Active subset / preprocessed features]",
        f"Active vocabulary classes: {report['active_vocabulary']['class_count']}",
        f"Classes with features: {report['preprocessing']['class_count_preprocessed']}",
        f"Excluded selected clips: {report['preprocessing']['excluded_clip_count']}",
        f"Exclusion reasons: {report['preprocessing']['exclusion_counts']}",
    ]
    replay_source_count = sum(
        result["source_samples"]
        for result in report["preprocessing_replay"].values()
    )
    replayed_count = sum(
        result["replayed_samples"]
        for result in report["preprocessing_replay"].values()
    )
    replay_issue_count = sum(
        result["missing_landmarks"]
        + result["mismatched_samples"]
        + result["invalid_sequences"]
        for result in report["preprocessing_replay"].values()
    )
    lines.extend(
        [
            f"Landmark-to-feature replay: {replayed_count}/{replay_source_count}; "
            f"issues={replay_issue_count}",
            "",
            "[Self-recorded samples]",
        ]
    )
    self_recorded = report["self_recorded_data"]
    lines.append(
        "Database: "
        + (
            f"{self_recorded['sample_count']} samples; "
            f"statuses={self_recorded['status_counts']}"
            if self_recorded["available"]
            else self_recorded["reason"]
        )
    )
    if self_recorded["available"]:
        lines.extend(
            [
                f"Unique target labels/signers: "
                f"{self_recorded['unique_target_labels']}/"
                f"{self_recorded['unique_nonempty_signers']}",
                f"Target labels: {', '.join(self_recorded['target_labels']) or 'none'}",
                f"Replayable sequences: "
                f"{self_recorded['replayable_sample_count']}/"
                f"{self_recorded['sample_count']}; "
                f"missing metadata={self_recorded['missing_metadata_counts']}",
                f"Accepted without verified ground truth: "
                f"{self_recorded['accepted_without_ground_truth_count']}",
                f"Missing/invalid sequence data: "
                f"{self_recorded['missing_sequence_count']}/"
                f"{self_recorded['invalid_sequence_count']}",
            ]
        )
    lines.append("")
    for split in _SPLITS:
        subset = report["active_subset"][split]
        features = report["features"][split]
        lines.extend(
            [
                f"{split}: subset={subset['sample_count']} clips/"
                f"{subset['class_count']} classes; features={features['samples']} "
                f"samples/{len(features['classes'])} classes",
                "  Feature class counts: "
                + ", ".join(
                    f"{label}={count}"
                    for label, count in sorted(features["class_counts"].items())
                ),
            ]
        )
    lines.extend(
        [
            "",
            "[Selected checkpoint and UI]",
            f"Checkpoint: {report['selected_checkpoint']['path']}",
            f"Checkpoint classes/output dimension: "
            f"{report['selected_checkpoint']['class_count']}/"
            f"{report['selected_checkpoint']['output_count']}",
            f"Explicit checkpoint class mapping present: "
            f"{report['selected_checkpoint']['explicit_class_mapping_present']}",
            f"Testing and Demo classes: "
            f"{len(report['ui_vocabulary']['testing_mode_class_names'])} "
            "(same checkpoint vocabulary)",
        ]
    )
    evaluation = report["selected_checkpoint"]["evaluation"]
    if evaluation["available"]:
        lines.extend(
            [
                f"Official test-set top-1: {evaluation['top1_correct']}/"
                f"{evaluation['test_samples']} "
                f"({evaluation['top1_accuracy']:.1%})",
                f"Official test-set top-5: {evaluation['top5_correct']}/"
                f"{evaluation['test_samples']} "
                f"({evaluation['top5_accuracy']:.1%})",
                f"Official test-set macro-F1: {evaluation['macro_f1']:.3f}",
                evaluation["note"],
            ]
        )
    else:
        lines.append(f"Checkpoint evaluation unavailable: {evaluation['reason']}")
    signer_report = report["signer_independence"]
    lines.append(
        "Subset signer overlap: "
        + (
            "none"
            if signer_report["available"]
            and all(not values for values in signer_report["overlap"].values())
            else str(signer_report["overlap"])
        )
    )
    low_sample_labels = [
        row["label"]
        for row in report["class_coverage"]
        if "LOW_SAMPLE_COUNT" in row["statuses"]
    ]
    lines.extend(
        [
            f"Active classes below sample thresholds: "
            f"{', '.join(low_sample_labels) if low_sample_labels else 'none'}",
            "",
            "[Class status / per-class split counts]",
            "Label | train | val | test | status",
        ]
    )
    active_labels = set(report["active_vocabulary"]["class_names"])
    for row in report["class_coverage"]:
        if row["label"] not in active_labels:
            continue
        counts = row["feature_sample_counts"]
        lines.append(
            f"{row['label']} | {counts['train']} | {counts['val']} | "
            f"{counts['test']} | {','.join(row['statuses'])}"
        )
    lines.extend(
        [
            "",
            "[Full machine-readable class lists]",
            "See the sibling JSON report for exhaustive raw, selected, processed, "
            "training, checkpoint, and UI class-name lists and per-clip exclusion reasons.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=C.MODELS / "gru_seed3.pt",
        help="checkpoint used for Testing/Demo Mode coverage",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=C.REPORTS / "dataset_coverage_report.json",
        help="JSON report output path",
    )
    args = parser.parse_args()
    report = audit_dataset(args.checkpoint)
    output_path = args.output
    if not output_path.is_absolute():
        output_path = C.ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    text_path = output_path.with_suffix(".txt")
    text_path.write_text(_render_text(report), encoding="utf-8")
    print(_render_text(report))
    print(f"JSON report saved to {output_path}")
    print(f"Text report saved to {text_path}")


if __name__ == "__main__":
    main()
