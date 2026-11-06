"""Persistence and split-safe export for self-recorded webcam samples."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import config as C
from dataset import FEATURE_DIM, build_features, preprocess_sequence
from evaluation_sequences import load_attempt_sequence
from model_vocabulary import build_class_mapping


_SPLITS = {
    "unassigned",
    "train",
    "validation",
    "test",
    "final_test",
    "real_world_eval",
}
_LOCKED_SPLITS = {"final_test", "real_world_eval"}
_SPLIT_FAMILY = {"train", "validation", "test", "final_test", "real_world_eval"}
_MIN_RAW_FRAMES = 8
_MIN_HAND_FRAMES = 3
_HAND_WARNING_RATIO = 0.4
_FEATURE_DIM = FEATURE_DIM
_CAMERA_ANGLES = {"front", "slightly left", "slightly right", "unknown"}
_DISTANCES = {"near", "normal", "far", "unknown"}
_LIGHTING = {"bright", "normal", "dim", "unknown"}
_SIGNING_SPEEDS = {"slow", "normal", "fast", "unknown"}
_DOMINANT_HANDS = {"RIGHT", "LEFT", "UNKNOWN"}
_SIGNER_ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,30}\d")
_TRAINING_SPLITS = {"train", "validation", "val", "test"}


def _detection_metrics(raw_landmarks: np.ndarray) -> dict[str, int | float]:
    pose_frames = np.isfinite(raw_landmarks[:, C.POSE_SL, :]).all(axis=(1, 2))
    shoulder_frames = np.isfinite(
        raw_landmarks[:, [C.L_SH, C.R_SH], :]
    ).all(axis=(1, 2))
    return {
        "pose_frame_count": int(pose_frames.sum()),
        "pose_frame_ratio": float(pose_frames.mean()),
        "shoulder_frame_count": int(shoulder_frames.sum()),
        "shoulder_frame_ratio": float(shoulder_frames.mean()),
    }


def _sequence_sha256(raw_landmarks: np.ndarray, model_features: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(raw_landmarks, dtype=np.float32).tobytes()
        + np.ascontiguousarray(model_features, dtype=np.float32).tobytes()
    ).hexdigest()


def validate_recording_arrays(
    raw_landmarks: np.ndarray,
    normalized_positions: np.ndarray,
    hand_present: np.ndarray,
    model_features: np.ndarray,
) -> tuple[int, float, list[str]]:
    if raw_landmarks.ndim != 3 or raw_landmarks.shape[1:] != (C.N_POINTS, 3):
        raise ValueError(
            f"Raw landmarks must have shape [frames, {C.N_POINTS}, 3]."
        )
    if len(raw_landmarks) < _MIN_RAW_FRAMES:
        raise ValueError(
            f"Sequence too short: {len(raw_landmarks)} usable frames; "
            f"at least {_MIN_RAW_FRAMES} are required."
        )
    if normalized_positions.shape != (C.SEQ_LEN, C.N_POINTS, 3):
        raise ValueError(
            f"Processed positions must have shape [{C.SEQ_LEN}, {C.N_POINTS}, 3]."
        )
    if hand_present.shape != (C.SEQ_LEN, 2):
        raise ValueError(f"Hand-presence flags must have shape [{C.SEQ_LEN}, 2].")
    if model_features.shape != (C.SEQ_LEN, _FEATURE_DIM):
        raise ValueError(
            f"Model features must have shape [{C.SEQ_LEN}, {_FEATURE_DIM}]."
        )
    if not np.isfinite(normalized_positions).all() or not np.isfinite(model_features).all():
        raise ValueError("Processed sequence contains invalid or non-finite values.")
    if np.isinf(raw_landmarks).any():
        raise ValueError("Raw landmark sequence contains infinite values.")
    complete_points = np.isfinite(raw_landmarks).all(axis=2)
    missing_points = np.isnan(raw_landmarks).all(axis=2)
    if not (complete_points | missing_points).all():
        raise ValueError(
            "Raw landmarks contain partially missing point coordinates."
        )
    if not np.isin(hand_present, (0, 1)).all():
        raise ValueError("Hand-presence flags must contain only zero or one.")
    expected_positions, expected_present = preprocess_sequence(raw_landmarks)
    expected_features = build_features(expected_positions, expected_present)
    if not np.array_equal(
        expected_positions, normalized_positions, equal_nan=True
    ):
        raise ValueError("Saved normalized positions do not match preprocessing.")
    if not np.array_equal(expected_present, hand_present, equal_nan=True):
        raise ValueError("Saved hand-presence flags do not match preprocessing.")
    if not np.array_equal(expected_features, model_features, equal_nan=True):
        raise ValueError("Saved model features do not match preprocessing.")

    raw_hand_present = (
        ~np.isnan(raw_landmarks[:, C.LH, 0]).any(axis=1)
        | ~np.isnan(raw_landmarks[:, C.RH, 0]).any(axis=1)
    )
    hand_frames = int(raw_hand_present.sum())
    if hand_frames < _MIN_HAND_FRAMES:
        raise ValueError(
            f"Hand landmarks were found in only {hand_frames} frames; "
            f"at least {_MIN_HAND_FRAMES} are needed."
        )
    ratio = hand_frames / len(raw_landmarks)
    warnings = []
    if ratio < _HAND_WARNING_RATIO:
        warnings.append(
            f"Hand landmarks were detected in {ratio:.0%} of captured frames. "
            "Review the sequence; temporary detection gaps are expected."
        )
    return hand_frames, ratio, warnings


class SelfRecordedSampleStore:
    """Keep collection candidates separate from public features and training data."""

    def __init__(self, database_path: Path | None = None) -> None:
        self.root = C.SELF_RECORDING
        self.database_path = database_path or self.root / "samples.sqlite3"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        for directory in ("staging", "accepted", "rejected", "raw"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS self_recorded_samples (
                    sample_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending_review', 'accepted', 'rejected')
                    ),
                    target_label TEXT NOT NULL,
                    ground_truth_label TEXT,
                    reviewer_id TEXT NOT NULL DEFAULT '',
                    signer_id TEXT NOT NULL DEFAULT '',
                    camera_index INTEGER NOT NULL DEFAULT 0,
                    camera_angle TEXT NOT NULL DEFAULT 'unknown',
                    distance TEXT NOT NULL DEFAULT 'unknown',
                    lighting TEXT NOT NULL DEFAULT 'unknown',
                    signing_speed TEXT NOT NULL DEFAULT 'unknown',
                    notes TEXT NOT NULL DEFAULT '',
                    recorded_at TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    original_frame_count INTEGER NOT NULL,
                    sequence_length INTEGER NOT NULL,
                    feature_dimension INTEGER NOT NULL,
                    hand_frame_count INTEGER NOT NULL,
                    hand_frame_ratio REAL NOT NULL,
                    pose_frame_count INTEGER,
                    pose_frame_ratio REAL,
                    shoulder_frame_count INTEGER,
                    shoulder_frame_ratio REAL,
                    quality_warnings_json TEXT NOT NULL DEFAULT '[]',
                    preprocessing_version TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    source_id TEXT NOT NULL UNIQUE,
                    sequence_sha256 TEXT NOT NULL UNIQUE,
                    sequence_path TEXT NOT NULL,
                    raw_video_path TEXT NOT NULL DEFAULT '',
                    dataset_split TEXT NOT NULL DEFAULT 'unassigned'
                        CHECK (dataset_split IN (
                            'unassigned', 'train', 'validation', 'test',
                            'final_test', 'real_world_eval'
                        )),
                    review_notes TEXT NOT NULL DEFAULT ''
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(self_recorded_samples)"
                )
            }
            if "reviewer_id" not in columns:
                connection.execute(
                    "ALTER TABLE self_recorded_samples ADD COLUMN "
                    "reviewer_id TEXT NOT NULL DEFAULT ''"
                )
            if "camera_index" not in columns:
                connection.execute(
                    "ALTER TABLE self_recorded_samples ADD COLUMN "
                    "camera_index INTEGER NOT NULL DEFAULT 0"
                )
            for name, declaration in (
                ("session_id", "TEXT"),
                ("dominant_hand", "TEXT"),
                ("camera_source", "TEXT"),
                ("recording_status", "TEXT"),
                ("verification_status", "TEXT"),
                ("quality_status", "TEXT"),
                ("rejection_reason", "TEXT"),
                ("pose_frame_count", "INTEGER"),
                ("pose_frame_ratio", "REAL"),
                ("shoulder_frame_count", "INTEGER"),
                ("shoulder_frame_ratio", "REAL"),
            ):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE self_recorded_samples ADD COLUMN "
                        f"{name} {declaration}"
                    )
            connection.execute(
                """
                UPDATE self_recorded_samples
                SET recording_status = CASE status
                    WHEN 'pending_review' THEN 'PENDING_REVIEW'
                    WHEN 'rejected' THEN 'REJECTED'
                    WHEN 'accepted' THEN 'ACCEPTED_UNVERIFIED'
                    ELSE NULL
                END
                WHERE recording_status IS NULL
                """
            )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def new_sample_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def new_session_id() -> str:
        return f"session-{uuid.uuid4().hex}"

    @staticmethod
    def validate_signer_id(signer_id: str) -> str:
        normalized = signer_id.strip()
        if _SIGNER_ID_PATTERN.fullmatch(normalized) is None:
            raise ValueError(
                "Enter an anonymous signer ID (for example S001); "
                "real names and blank IDs are not allowed."
            )
        return normalized

    def signer_ids(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT signer_id
                FROM self_recorded_samples
                WHERE signer_id IS NOT NULL AND trim(signer_id) != ''
                ORDER BY signer_id
                """
            ).fetchall()
        return [str(row["signer_id"]) for row in rows]

    @staticmethod
    def _legacy_recording_status(status: str) -> str | None:
        return {
            "pending_review": "PENDING_REVIEW",
            "rejected": "REJECTED",
            "accepted": "ACCEPTED_UNVERIFIED",
        }.get(status)

    def _write_sequence_metadata(
        self, sequence_path: Path, row: sqlite3.Row | dict[str, Any]
    ) -> None:
        sequence_path = self.resolve_artifact_path(sequence_path)
        try:
            with np.load(sequence_path, allow_pickle=False) as archive:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in (
                        "raw_landmarks",
                        "normalized_positions",
                        "hand_present",
                        "model_features",
                    )
                }
                previous = json.loads(str(archive["metadata"].item()))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Cannot synchronize sequence metadata in {sequence_path}."
            ) from exc
        previous.update(
            {
                "sample_id": row["sample_id"],
                "target_label": row["target_label"],
                "ground_truth_label": row["ground_truth_label"],
                "signer_id": row["signer_id"] or None,
                "session_id": row["session_id"],
                "recorded_at": row["recorded_at"],
                "dominant_hand": row["dominant_hand"],
                "camera_source": row["camera_source"],
                "recording_status": (
                    row["recording_status"]
                    or self._legacy_recording_status(row["status"])
                ),
                "verification_status": row["verification_status"],
                "quality_status": row["quality_status"],
                "rejection_reason": row["rejection_reason"],
                "pose_frame_count": row["pose_frame_count"],
                "pose_frame_ratio": row["pose_frame_ratio"],
                "shoulder_frame_count": row["shoulder_frame_count"],
                "shoulder_frame_ratio": row["shoulder_frame_ratio"],
                "duration_seconds": row["duration_seconds"],
                "original_frame_count": row["original_frame_count"],
                "sequence_length": row["sequence_length"],
                "feature_dimension": row["feature_dimension"],
                "hand_frame_count": row["hand_frame_count"],
                "hand_frame_ratio": row["hand_frame_ratio"],
                "quality_warnings": json.loads(row["quality_warnings_json"]),
                "sequence_path": str(sequence_path.resolve()),
                "raw_video_path": row["raw_video_path"] or None,
                "dataset_split": row["dataset_split"],
                "preprocessing_version": row["preprocessing_version"],
                "model_version": row["model_version"],
                "camera_angle": row["camera_angle"],
                "camera_index": row["camera_index"],
                "distance": row["distance"],
                "lighting": row["lighting"],
                "signing_speed": row["signing_speed"],
                "notes": row["notes"],
            }
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=sequence_path.parent,
                prefix=f".{sequence_path.stem}.",
                suffix=".npz",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary_path,
                raw_landmarks=arrays["raw_landmarks"],
                normalized_positions=arrays["normalized_positions"],
                hand_present=arrays["hand_present"],
                model_features=arrays["model_features"],
                metadata=np.asarray(
                    json.dumps(previous, ensure_ascii=False, allow_nan=False)
                ),
            )
            os.replace(temporary_path, sequence_path)
        except OSError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    def resolve_artifact_path(
        self, stored_path: str | Path, *, must_exist: bool = True
    ) -> Path:
        """Resolve a saved artifact path without allowing access outside this store."""
        candidate = Path(stored_path).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ValueError("Recording artifact path is outside self-recorded storage.") from exc
        if must_exist and not candidate.is_file():
            raise FileNotFoundError(f"Recording artifact not found: {candidate}")
        return candidate

    def _load_valid_sequence(
        self, sample: sqlite3.Row, *, require_metadata_match: bool = False
    ) -> dict[str, Any]:
        sequence = load_attempt_sequence(
            self.resolve_artifact_path(sample["sequence_path"])
        )
        validate_recording_arrays(
            sequence["raw_landmarks"],
            sequence["normalized_positions"],
            sequence["hand_present"],
            sequence["model_features"],
        )
        if (
            _sequence_sha256(
                sequence["raw_landmarks"], sequence["model_features"]
            )
            != sample["sequence_sha256"]
        ):
            raise ValueError(
                f"Sample {sample['sample_id']} sequence hash does not match SQLite."
            )
        if require_metadata_match:
            metadata = sequence.get("metadata", {})
            for field in (
                "sample_id",
                "target_label",
                "ground_truth_label",
                "signer_id",
                "session_id",
                "recording_status",
                "verification_status",
                "quality_status",
                "dataset_split",
            ):
                if metadata.get(field) != sample[field]:
                    raise ValueError(
                        f"Sample {sample['sample_id']} sequence metadata "
                        f"{field} does not match SQLite."
                    )
        return sequence

    def create_pending(
        self,
        *,
        sample_id: str,
        target_label: str,
        signer_id: str = "",
        session_id: str = "",
        dominant_hand: str = "UNKNOWN",
        camera_source: str = "",
        camera_index: int = 0,
        camera_angle: str = "unknown",
        distance: str = "unknown",
        lighting: str = "unknown",
        signing_speed: str = "unknown",
        notes: str = "",
        duration_seconds: float,
        original_frame_count: int,
        recorded_at: str | None = None,
        raw_landmarks: np.ndarray,
        normalized_positions: np.ndarray,
        hand_present: np.ndarray,
        model_features: np.ndarray,
        model_version: str,
        raw_video_path: str = "",
    ) -> dict[str, Any]:
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", sample_id) is None
            or not target_label.strip()
        ):
            raise ValueError("A unique sample ID and target label are required.")
        signer_id = self.validate_signer_id(signer_id)
        if not session_id.strip():
            raise ValueError("A recording session ID is required.")
        if dominant_hand not in _DOMINANT_HANDS:
            raise ValueError("Dominant hand must be RIGHT, LEFT, or UNKNOWN.")
        if not camera_source.strip():
            raise ValueError("Camera source metadata is required.")
        if (
            isinstance(camera_index, bool)
            or not isinstance(camera_index, int)
            or camera_index < 0
        ):
            raise ValueError("Camera index must be a non-negative integer.")
        for field, value, options in (
            ("camera angle", camera_angle, _CAMERA_ANGLES),
            ("distance", distance, _DISTANCES),
            ("lighting", lighting, _LIGHTING),
            ("signing speed", signing_speed, _SIGNING_SPEEDS),
        ):
            if value not in options:
                raise ValueError(f"Unsupported {field}: {value}")
        if (
            isinstance(duration_seconds, bool)
            or not math.isfinite(duration_seconds)
            or duration_seconds <= 0
        ):
            raise ValueError("Recording duration must be finite and positive.")
        if (
            isinstance(original_frame_count, bool)
            or not isinstance(original_frame_count, int)
            or original_frame_count <= 0
        ):
            raise ValueError("Original frame count must be a positive integer.")
        if original_frame_count != len(raw_landmarks):
            raise ValueError(
                "Original frame count must match the saved raw landmark sequence."
            )
        if not model_version.strip():
            raise ValueError("The checkpoint version must be recorded.")
        hand_frame_count, ratio, warnings = validate_recording_arrays(
            raw_landmarks,
            normalized_positions,
            hand_present,
            model_features,
        )
        detection = _detection_metrics(raw_landmarks)
        if detection["pose_frame_count"] == 0:
            warnings.append(
                "No upper-body pose landmarks were detected; shoulder-based "
                "normalization used the preprocessing fallback."
            )
        if detection["shoulder_frame_count"] == 0:
            warnings.append(
                "Shoulder landmarks were not detected; inspect normalized pose "
                "coordinates before accepting this sample."
            )
        sequence_hash = _sequence_sha256(raw_landmarks, model_features)
        source_id = sample_id.strip()
        if recorded_at is None:
            recorded_at = datetime.now(timezone.utc).isoformat()
        else:
            try:
                parsed_recorded_at = datetime.fromisoformat(recorded_at)
            except ValueError as exc:
                raise ValueError("Recording time must be an ISO 8601 timestamp.") from exc
            if parsed_recorded_at.tzinfo is None:
                raise ValueError("Recording time must include a timezone.")
        sequence_path = self.root / "staging" / f"{sample_id}.npz"
        resolved_video_path = ""
        if raw_video_path:
            video = Path(raw_video_path).resolve()
            try:
                video.relative_to((self.root / "raw").resolve())
            except ValueError as exc:
                raise ValueError(
                    "Optional raw video must be saved inside self-recorded raw storage."
                ) from exc
            if not video.is_file():
                raise FileNotFoundError(f"Optional raw recording was not saved: {video}")
            resolved_video_path = str(video)
        metadata = {
            "sample_id": sample_id,
            "target_label": target_label.strip(),
            "ground_truth_label": None,
            "signer_id": signer_id,
            "session_id": session_id.strip(),
            "recorded_at": recorded_at,
            "dominant_hand": dominant_hand,
            "camera_source": camera_source.strip(),
            "recording_status": "PENDING_REVIEW",
            "verification_status": "UNVERIFIED",
            "quality_status": "PASSED",
            "rejection_reason": None,
            **detection,
            "sequence_path": str(sequence_path.resolve()),
            "raw_video_path": resolved_video_path or None,
            "dataset_split": "unassigned",
            "camera_index": camera_index,
            "camera_angle": camera_angle,
            "distance": distance,
            "lighting": lighting,
            "signing_speed": signing_speed,
            "notes": notes.strip(),
            "preprocessing_version": C.PREPROCESSING_VERSION,
            "model_version": model_version,
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=sequence_path.parent,
                prefix=f".{sample_id}.",
                suffix=".npz",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary_path,
                raw_landmarks=raw_landmarks.astype(np.float32, copy=False),
                normalized_positions=normalized_positions.astype(np.float32, copy=False),
                hand_present=hand_present.astype(np.float32, copy=False),
                model_features=model_features.astype(np.float32, copy=False),
                metadata=np.asarray(
                    json.dumps(metadata, ensure_ascii=False, allow_nan=False)
                ),
            )
            os.replace(temporary_path, sequence_path)
        except OSError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO self_recorded_samples (
                        sample_id, status, target_label, signer_id, camera_angle,
                        camera_index, distance, lighting, signing_speed, notes, recorded_at,
                        duration_seconds, original_frame_count, sequence_length,
                        feature_dimension, hand_frame_count, hand_frame_ratio,
                        pose_frame_count, pose_frame_ratio,
                        shoulder_frame_count, shoulder_frame_ratio,
                        quality_warnings_json, preprocessing_version, model_version,
                        source_id, sequence_sha256, sequence_path, raw_video_path,
                        session_id, dominant_hand, camera_source, recording_status,
                        verification_status, quality_status, rejection_reason
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        sample_id,
                        "pending_review",
                        target_label.strip(),
                        signer_id,
                        camera_angle,
                        camera_index,
                        distance,
                        lighting,
                        signing_speed,
                        notes.strip(),
                        recorded_at,
                        float(duration_seconds),
                        int(original_frame_count),
                        C.SEQ_LEN,
                        _FEATURE_DIM,
                        hand_frame_count,
                        ratio,
                        detection["pose_frame_count"],
                        detection["pose_frame_ratio"],
                        detection["shoulder_frame_count"],
                        detection["shoulder_frame_ratio"],
                        json.dumps(warnings, ensure_ascii=False),
                        C.PREPROCESSING_VERSION,
                        model_version,
                        source_id,
                        sequence_hash,
                        str(sequence_path.resolve()),
                        resolved_video_path,
                        session_id.strip(),
                        dominant_hand,
                        camera_source.strip(),
                        "PENDING_REVIEW",
                        "UNVERIFIED",
                        "PASSED",
                        None,
                    ),
                )
        except sqlite3.Error:
            sequence_path.unlink(missing_ok=True)
            raise
        return {
            "sample_id": sample_id,
            "sequence_path": str(sequence_path.resolve()),
            "hand_frame_count": hand_frame_count,
            "hand_frame_ratio": ratio,
            "quality_warnings": warnings,
        }

    def review(
        self,
        sample_id: str,
        decision: str,
        notes: str = "",
        rejection_reason: str = "",
    ) -> sqlite3.Row:
        if decision not in {"accepted", "rejected"}:
            raise ValueError("Review decision must be accepted or rejected.")
        if decision == "rejected" and not (
            rejection_reason.strip() or notes.strip()
        ):
            raise ValueError(
                "A rejected recording must include a clear rejection reason."
            )
        with self._connect() as connection:
            sample = connection.execute(
                "SELECT * FROM self_recorded_samples WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()
            if sample is None:
                raise KeyError(f"Self-recorded sample {sample_id} does not exist.")
            if sample["status"] != "pending_review":
                raise ValueError(f"Sample {sample_id} is already {sample['status']}.")
            source = self.resolve_artifact_path(
                sample["sequence_path"], must_exist=False
            )
            destination = self.root / decision / source.name
            if not source.is_file():
                if decision == "accepted":
                    raise FileNotFoundError(f"Sample sequence is missing: {source}")
                connection.execute(
                    """
                    UPDATE self_recorded_samples
                    SET status = 'rejected', review_notes = ?,
                        recording_status = 'REJECTED', rejection_reason = ?
                    WHERE sample_id = ?
                    """,
                    (
                        notes.strip(),
                        rejection_reason.strip() or notes.strip(),
                        sample_id,
                    ),
                )
                updated = connection.execute(
                    "SELECT * FROM self_recorded_samples WHERE sample_id = ?",
                    (sample_id,),
                ).fetchone()
                if updated is None:
                    raise KeyError(f"Self-recorded sample {sample_id} does not exist.")
                return updated
            if decision == "accepted":
                self._load_valid_sequence(sample)
            video_source = (
                self.resolve_artifact_path(sample["raw_video_path"])
                if sample["raw_video_path"]
                else None
            )
            video_destination = self.root / decision / (
                f"{sample_id}{video_source.suffix}" if video_source else ""
            )
            if destination.exists() or (
                video_source is not None and video_destination.exists()
            ):
                raise FileExistsError(
                    f"Review destination already exists for sample {sample_id}."
                )
        try:
            source.replace(destination)
            if video_source is not None:
                video_source.replace(video_destination)
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE self_recorded_samples
                    SET status = ?, review_notes = ?, sequence_path = ?,
                        raw_video_path = ?, recording_status = ?,
                        rejection_reason = ?
                    WHERE sample_id = ?
                    """,
                    (
                        decision,
                        notes.strip(),
                        str(destination.resolve()),
                        str(video_destination.resolve()) if video_source else "",
                        (
                            "REJECTED"
                            if decision == "rejected"
                            else "ACCEPTED_UNVERIFIED"
                        ),
                        (
                            rejection_reason.strip() or notes.strip() or None
                            if decision == "rejected"
                            else None
                        ),
                        sample_id,
                    ),
                )
                updated = connection.execute(
                    "SELECT * FROM self_recorded_samples WHERE sample_id = ?",
                    (sample_id,),
                ).fetchone()
                if updated is None:
                    raise KeyError(f"Self-recorded sample {sample_id} does not exist.")
                self._write_sequence_metadata(destination, updated)
        except (sqlite3.Error, OSError, ValueError, TypeError, KeyError):
            if destination.exists():
                destination.replace(source)
            if video_source and video_destination.exists():
                video_destination.replace(video_source)
            if source.is_file():
                self._write_sequence_metadata(source, sample)
            raise
        return updated

    def verify_ground_truth(
        self,
        sample_id: str,
        label: str,
        reviewer_id: str,
        reviewer_notes: str = "",
    ) -> None:
        if not label.strip():
            raise ValueError("Verified ground-truth label must not be empty.")
        if not reviewer_id.strip():
            raise ValueError("An anonymous reviewer ID is required for verification.")
        sample = self.get(sample_id)
        if sample["status"] != "accepted":
            raise ValueError("Only accepted recordings can receive verified ground truth.")
        self._load_valid_sequence(sample)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE self_recorded_samples
                    SET ground_truth_label = ?, reviewer_id = ?, review_notes = ?,
                        recording_status = 'VERIFIED',
                        verification_status = 'VERIFIED', quality_status = 'PASSED'
                    WHERE sample_id = ? AND status = 'accepted'
                    """,
                    (
                        label.strip(),
                        reviewer_id.strip(),
                        reviewer_notes.strip(),
                        sample_id,
                    ),
                )
                updated = connection.execute(
                    "SELECT * FROM self_recorded_samples WHERE sample_id = ?",
                    (sample_id,),
                ).fetchone()
                if updated is None:
                    raise KeyError(f"Self-recorded sample {sample_id} does not exist.")
                self._write_sequence_metadata(
                    self.resolve_artifact_path(updated["sequence_path"]), updated
                )
        except (sqlite3.Error, OSError, ValueError, TypeError, KeyError):
            self._write_sequence_metadata(
                self.resolve_artifact_path(sample["sequence_path"]), sample
            )
            raise

    def assign_split(self, sample_id: str, dataset_split: str) -> None:
        if dataset_split not in _SPLITS:
            raise ValueError(f"Unsupported dataset split: {dataset_split}")
        with self._connect() as connection:
            sample = connection.execute(
                "SELECT * FROM self_recorded_samples WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()
            if sample is None:
                raise KeyError(f"Self-recorded sample {sample_id} does not exist.")
            if (
                sample["status"] != "accepted"
                or sample["recording_status"] != "VERIFIED"
                or sample["verification_status"] != "VERIFIED"
                or not sample["ground_truth_label"]
                or sample["quality_status"] != "PASSED"
            ):
                raise ValueError(
                    "Only accepted recordings with explicitly verified ground truth "
                    "and passed quality checks can be assigned a dataset split."
                )
            if not sample["signer_id"].strip():
                raise ValueError(
                    "An anonymous signer ID is required before assigning a dataset "
                    "split so signer leakage can be prevented."
                )
            if not sample["session_id"]:
                raise ValueError(
                    "A recording session ID is required before assigning a dataset split."
                )
            self._load_valid_sequence(sample, require_metadata_match=True)
            current_split = sample["dataset_split"]
            if current_split in _LOCKED_SPLITS and dataset_split != current_split:
                raise ValueError(
                    f"{current_split} samples are locked and cannot be reassigned."
                )
            if dataset_split != "unassigned":
                conflicts = connection.execute(
                    """
                    SELECT sample_id, dataset_split, signer_id, source_id,
                           sequence_sha256
                    FROM self_recorded_samples
                    WHERE sample_id != ? AND dataset_split != 'unassigned'
                      AND (
                        source_id = ? OR sequence_sha256 = ?
                        OR (
                            signer_id != '' AND signer_id = ?
                            AND dataset_split != ?
                            AND dataset_split IN (
                                'train', 'validation', 'test', 'final_test',
                                'real_world_eval'
                            )
                        )
                      )
                    """,
                    (
                        sample_id,
                        sample["source_id"],
                        sample["sequence_sha256"],
                        sample["signer_id"],
                        dataset_split,
                    ),
                ).fetchall()
                if conflicts:
                    conflict = conflicts[0]
                    if (
                        conflict["source_id"] == sample["source_id"]
                        or conflict["sequence_sha256"] == sample["sequence_sha256"]
                    ):
                        raise ValueError(
                            "This source recording or derived sequence is already "
                            f"assigned to {conflict['dataset_split']}."
                        )
                    raise ValueError(
                        f"Signer {sample['signer_id']} already appears in "
                        f"{conflict['dataset_split']}; signer overlap is forbidden."
                    )
            connection.execute(
                "UPDATE self_recorded_samples SET dataset_split = ? WHERE sample_id = ?",
                (dataset_split, sample_id),
            )
            updated = connection.execute(
                "SELECT * FROM self_recorded_samples WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()
            if updated is not None:
                self._write_sequence_metadata(
                    self.resolve_artifact_path(updated["sequence_path"]), updated
                )

    def get(self, sample_id: str) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM self_recorded_samples WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Self-recorded sample {sample_id} does not exist.")
        return row

    def samples(self, *, status: str | None = None) -> list[sqlite3.Row]:
        query = "SELECT * FROM self_recorded_samples"
        parameters: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY recorded_at, sample_id"
        with self._connect() as connection:
            return connection.execute(query, parameters).fetchall()

    def dashboard(self, vocabulary: list[str]) -> dict[str, Any]:
        accepted = self.samples(status="accepted")
        rejected = self.samples(status="rejected")
        by_sign: dict[str, dict[str, Any]] = {
            label: {"samples": 0, "signers": set()} for label in vocabulary
        }
        for sample in accepted:
            row = by_sign.setdefault(
                sample["target_label"], {"samples": 0, "signers": set()}
            )
            row["samples"] += 1
            if sample["signer_id"]:
                row["signers"].add(sample["signer_id"])
        sign_rows = [
            {
                "label": label,
                "samples": row["samples"],
                "signers": len(row["signers"]),
            }
            for label, row in by_sign.items()
        ]
        sign_rows.sort(key=lambda row: (row["samples"], row["label"]))
        return {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "signers": len({row["signer_id"] for row in accepted if row["signer_id"]}),
            "classes_covered": sum(row["samples"] > 0 for row in sign_rows),
            "by_sign": sign_rows,
        }

    def export_split(self, dataset_split: str, output_dir: Path) -> int:
        if dataset_split not in _SPLIT_FAMILY:
            raise ValueError("Choose an explicit train, validation, test, final_test, "
                             "or real_world_eval split.")
        rows = [
            row for row in self.samples(status="accepted")
            if row["dataset_split"] == dataset_split
            and row["ground_truth_label"]
            and row["recording_status"] == "VERIFIED"
            and row["verification_status"] == "VERIFIED"
            and row["quality_status"] == "PASSED"
            and row["signer_id"].strip()
            and row["session_id"]
        ]
        if not rows:
            raise ValueError(f"No reviewed accepted samples are assigned to {dataset_split}.")
        for row in rows:
            self._load_valid_sequence(row, require_metadata_match=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.csv"
        fields = (
            "sample_id",
            "target_label",
            "ground_truth_label",
            "reviewer_id",
            "signer_id",
            "camera_index",
            "dataset_split",
            "camera_angle",
            "distance",
            "lighting",
            "signing_speed",
            "recorded_at",
            "duration_seconds",
            "notes",
            "review_notes",
            "recording_status",
            "verification_status",
            "quality_status",
            "rejection_reason",
            "session_id",
            "dominant_hand",
            "camera_source",
            "original_frame_count",
            "sequence_length",
            "feature_dimension",
            "hand_frame_count",
            "hand_frame_ratio",
            "pose_frame_count",
            "pose_frame_ratio",
            "shoulder_frame_count",
            "shoulder_frame_ratio",
            "preprocessing_version",
            "model_version",
            "source_id",
            "sequence_sha256",
            "sequence_path",
            "raw_video_path",
        )
        output_rows = []
        for row in rows:
            sequence_source = self.resolve_artifact_path(row["sequence_path"])
            sequence_name = f"{row['sample_id']}.npz"
            shutil.copy2(sequence_source, output_dir / sequence_name)
            video_name = ""
            if row["raw_video_path"]:
                video_source = self.resolve_artifact_path(row["raw_video_path"])
                video_name = video_source.name
                shutil.copy2(video_source, output_dir / video_name)
            item = {field: row[field] for field in fields}
            item["sequence_path"] = sequence_name
            item["raw_video_path"] = video_name
            output_rows.append(item)
        with manifest_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)
        return len(output_rows)

    def export_feature_archive(
        self,
        dataset_split: str,
        output_path: Path,
        vocabulary: list[str],
    ) -> dict[str, Any]:
        """Create a model-ready, provenance-preserving archive for one split."""
        if dataset_split not in _TRAINING_SPLITS:
            raise ValueError(
                "Feature integration supports only train, validation, or test."
            )
        archive_split = "val" if dataset_split == "validation" else dataset_split
        labels = {label: index for index, label in enumerate(vocabulary)}
        rows = [
            row
            for row in self.samples(status="accepted")
            if row["dataset_split"] == dataset_split
            and row["ground_truth_label"]
            and row["recording_status"] == "VERIFIED"
            and row["verification_status"] == "VERIFIED"
            and row["quality_status"] == "PASSED"
            and row["signer_id"]
            and row["session_id"]
        ]
        rows.sort(key=lambda row: row["sample_id"])
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate sample IDs found in the selected split.")

        positions: list[np.ndarray] = []
        presence: list[np.ndarray] = []
        class_indices: list[int] = []
        signers: list[str] = []
        sample_ids: list[str] = []
        targets: list[str] = []
        ground_truth: list[str] = []
        sessions: list[str] = []
        recorded_at: list[str] = []
        dominant_hands: list[str] = []
        camera_sources: list[str] = []
        camera_indices: list[int] = []
        camera_angles: list[str] = []
        distances: list[str] = []
        lighting: list[str] = []
        signing_speeds: list[str] = []
        sequence_hashes: list[str] = []
        for row in rows:
            label = str(row["ground_truth_label"])
            if label not in labels:
                raise ValueError(
                    f"Verified label {label!r} for {row['sample_id']} is not in "
                    "the model vocabulary."
                )
            sequence = self._load_valid_sequence(
                row, require_metadata_match=True
            )
            if row["sequence_length"] != C.SEQ_LEN or row["feature_dimension"] != _FEATURE_DIM:
                raise ValueError(
                    f"Sample {row['sample_id']} has incompatible feature dimensions."
                )
            positions.append(sequence["normalized_positions"])
            presence.append(sequence["hand_present"])
            class_indices.append(labels[label])
            signers.append(str(row["signer_id"]))
            sample_ids.append(str(row["sample_id"]))
            targets.append(str(row["target_label"]))
            ground_truth.append(label)
            sessions.append(str(row["session_id"]))
            recorded_at.append(str(row["recorded_at"]))
            dominant_hands.append(str(row["dominant_hand"] or "UNKNOWN"))
            camera_sources.append(str(row["camera_source"] or ""))
            camera_indices.append(
                -1 if row["camera_index"] is None else int(row["camera_index"])
            )
            camera_angles.append(str(row["camera_angle"] or "unknown"))
            distances.append(str(row["distance"] or "unknown"))
            lighting.append(str(row["lighting"] or "unknown"))
            signing_speeds.append(str(row["signing_speed"] or "unknown"))
            sequence_hashes.append(str(row["sequence_sha256"]))

        count = len(rows)
        pos_array = (
            np.stack(positions)
            if positions
            else np.empty((0, C.SEQ_LEN, C.N_POINTS, 3), dtype=np.float32)
        )
        present_array = (
            np.stack(presence)
            if presence
            else np.empty((0, C.SEQ_LEN, 2), dtype=np.float32)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent,
                prefix=f".{output_path.stem}.",
                suffix=".npz",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary_path,
                pos=pos_array,
                present=present_array,
                y=np.asarray(class_indices, dtype=np.int64),
                participant=np.asarray(
                    [f"self_recorded:{signer}" for signer in signers]
                ),
                video=np.asarray(
                    [f"self_recorded:{sample_id}" for sample_id in sample_ids]
                ),
                source=np.asarray(["self_recorded"] * count),
                sample_id=np.asarray(sample_ids),
                signer_id=np.asarray(signers),
                session_id=np.asarray(sessions),
                target_label=np.asarray(targets),
                ground_truth_label=np.asarray(ground_truth),
                dominant_hand=np.asarray(dominant_hands),
                camera_source=np.asarray(camera_sources),
                camera_index=np.asarray(camera_indices, dtype=np.int64),
                camera_angle=np.asarray(camera_angles),
                distance=np.asarray(distances),
                lighting=np.asarray(lighting),
                signing_speed=np.asarray(signing_speeds),
                recording_status=np.asarray(["VERIFIED"] * count),
                verification_status=np.asarray(["VERIFIED"] * count),
                quality_status=np.asarray(["PASSED"] * count),
                dataset_split=np.asarray([archive_split] * count),
                recorded_at=np.asarray(recorded_at),
                sequence_sha256=np.asarray(sequence_hashes),
                vocab=np.asarray(vocabulary),
                class_mapping_json=np.asarray(
                    json.dumps(build_class_mapping(vocabulary), ensure_ascii=False)
                ),
            )
            os.replace(temporary_path, output_path)
        except OSError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return {
            "split": archive_split,
            "samples": count,
            "class_counts": {
                vocabulary[index]: class_indices.count(index)
                for index in range(len(vocabulary))
                if class_indices.count(index)
            },
            "sample_ids": sample_ids,
            "output_path": str(output_path.resolve()),
        }
