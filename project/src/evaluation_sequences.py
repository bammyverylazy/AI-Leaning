"""Persistence helpers for isolated Testing Mode sequence artifacts."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import config as C


def save_attempt_sequence(
    attempt_id: int,
    *,
    raw_landmarks: np.ndarray,
    normalized_positions: np.ndarray,
    hand_present: np.ndarray,
    model_features: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    """Atomically save raw and exact model inputs outside the training data tree."""
    if attempt_id <= 0:
        raise ValueError("attempt_id must be positive")
    if raw_landmarks.ndim != 3 or raw_landmarks.shape[0] != C.SEQ_LEN:
        raise ValueError("raw landmark sequence must have shape [64, points, 3]")
    if raw_landmarks.shape[2] != 3:
        raise ValueError("raw landmarks must contain XYZ coordinates")
    if normalized_positions.ndim != 3 or normalized_positions.shape[0] != C.SEQ_LEN:
        raise ValueError("normalized positions must have shape [64, points, 3]")
    if hand_present.shape != (C.SEQ_LEN, 2):
        raise ValueError("hand-presence sequence must have shape [64, 2]")
    if model_features.ndim != 2 or model_features.shape[0] != C.SEQ_LEN:
        raise ValueError("model features must have shape [64, feature_dimension]")
    if not all(
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
        for array in (raw_landmarks, normalized_positions, hand_present, model_features)
    ):
        raise ValueError("sequence arrays must contain numeric values")
    encoded_metadata = json.dumps(
        metadata, ensure_ascii=False, allow_nan=False, sort_keys=True
    )
    C.TESTING_SEQUENCES.mkdir(parents=True, exist_ok=True)
    destination = C.TESTING_SEQUENCES / f"{attempt_id}.npz"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=C.TESTING_SEQUENCES,
            prefix=f".{attempt_id}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary_path,
            raw_landmarks=raw_landmarks,
            normalized_positions=normalized_positions,
            hand_present=hand_present,
            model_features=model_features,
            metadata=np.asarray(encoded_metadata),
        )
        os.replace(temporary_path, destination)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination.relative_to(C.ROOT).as_posix()


def resolve_sequence_path(relative_path: str) -> Path:
    """Resolve a database path while preventing reads outside evaluation storage."""
    candidate = (C.ROOT / relative_path).resolve()
    try:
        candidate.relative_to(C.TESTING_SEQUENCES.resolve())
    except ValueError as exc:
        raise ValueError("sequence artifact path is outside Testing Mode storage") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Testing sequence artifact not found: {candidate}")
    return candidate


def load_attempt_sequence(path: Path) -> dict[str, Any]:
    """Load and validate a sequence archive without allowing pickle objects."""
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "raw_landmarks",
            "normalized_positions",
            "hand_present",
            "model_features",
            "metadata",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Sequence archive is missing fields: {sorted(missing)}")
        result: dict[str, Any] = {
            key: np.array(archive[key], copy=True)
            for key in required.difference({"metadata"})
        }
        metadata_value = archive["metadata"]
        if metadata_value.ndim != 0:
            raise ValueError("Sequence metadata must be a scalar JSON value")
        result["metadata"] = json.loads(str(metadata_value.item()))
    model_features = result["model_features"]
    if model_features.ndim != 2 or model_features.shape[0] != C.SEQ_LEN:
        raise ValueError("Saved model feature sequence has an invalid shape")
    if (
        result["raw_landmarks"].ndim != 3
        or result["raw_landmarks"].shape[0] < 1
        or result["raw_landmarks"].shape[1:] != (C.N_POINTS, 3)
    ):
        raise ValueError("Saved raw landmark sequence has an invalid shape")
    if result["normalized_positions"].shape != (
        C.SEQ_LEN, C.N_POINTS, 3
    ):
        raise ValueError("Saved normalized positions have an invalid shape")
    if result["hand_present"].shape != (C.SEQ_LEN, 2):
        raise ValueError("Saved hand-presence sequence has an invalid shape")
    return result
