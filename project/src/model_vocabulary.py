"""Validation helpers for the ordered class mapping stored in model checkpoints."""
from __future__ import annotations

from typing import Any, cast

from label_display import canonical_label


def validate_vocabulary(labels: list[str] | tuple[str, ...]) -> list[str]:
    """Return labels in class-index order after validating unique class identities."""
    if not labels:
        raise ValueError("The model vocabulary must contain at least one label.")
    vocabulary = list(labels)
    if any(not isinstance(label, str) or not label.strip() for label in vocabulary):
        raise ValueError("Every model vocabulary label must be a non-empty string.")
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("The model vocabulary contains duplicate class labels.")

    canonical_labels = [canonical_label(label) for label in vocabulary]
    if len(set(canonical_labels)) != len(canonical_labels):
        duplicates = sorted(
            label
            for label in set(canonical_labels)
            if canonical_labels.count(label) > 1
        )
        raise ValueError(
            "The model vocabulary contains duplicate canonical labels: "
            + ", ".join(duplicates)
        )
    return vocabulary


def build_class_mapping(labels: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    """Build the explicit, stable class-index mapping serialized with new checkpoints."""
    vocabulary = validate_vocabulary(labels)
    return [
        {
            "class_index": index,
            "label": label,
            "canonical_label": canonical_label(label),
        }
        for index, label in enumerate(vocabulary)
    ]


def checkpoint_vocabulary(checkpoint: dict[str, Any]) -> list[str]:
    """Read and validate a checkpoint mapping, supporting legacy ``vocab`` checkpoints."""
    if "class_mapping" not in checkpoint:
        return validate_vocabulary(checkpoint["vocab"])

    mapping = checkpoint["class_mapping"]
    if not isinstance(mapping, list):
        raise ValueError("Checkpoint class_mapping must be a list.")
    if any(not isinstance(item, dict) for item in mapping):
        raise ValueError("Every checkpoint class_mapping entry must be an object.")
    entries = cast(list[dict[str, Any]], mapping)
    if any(
        isinstance(item.get("class_index"), bool)
        or not isinstance(item.get("class_index"), int)
        for item in entries
    ):
        raise ValueError("Checkpoint class_mapping indices must be integers.")
    ordered = sorted(entries, key=lambda item: item["class_index"])
    expected_indices = list(range(len(mapping)))
    indices = [item["class_index"] for item in ordered]
    if indices != expected_indices:
        raise ValueError(
            "Checkpoint class_mapping indices must be unique and contiguous from zero."
        )
    raw_labels = [item.get("label") for item in ordered]
    if any(not isinstance(label, str) for label in raw_labels):
        raise ValueError("Every checkpoint class_mapping label must be a string.")
    labels = validate_vocabulary(cast(list[str], raw_labels))
    expected_mapping = build_class_mapping(labels)
    if ordered != expected_mapping:
        raise ValueError(
            "Checkpoint class_mapping does not match its labels or canonical labels."
        )
    if list(checkpoint.get("vocab", [])) != labels:
        raise ValueError("Checkpoint vocab does not match class_mapping order.")
    return labels
