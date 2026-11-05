"""Canonical user-facing labels while preserving trained class identities."""
from __future__ import annotations

from dataclasses import dataclass

import torch

# These are dataset naming variants with distinct ASL-LEX identities. Their
# numeric suffix is meaningful, so keep it visible to distinguish the signs.
DISTINCT_VARIANT_LABELS = frozenset(
    {
        "AXE1", "BACKPACK1", "BASKETBALL1", "BEE1", "BELT1", "BITE1",
        "BREAKFAST1", "CANCER1", "CHRISTMAS1", "CHRISTMAS2", "CONFUSED1", "DARK1",
        "DEAF1", "DECIDE1", "DEMAND1", "DEVELOP1", "DINNER1", "DOG1",
        "DOWNSIZE1", "DRAG1", "EAT1", "EDIT1", "ELEVATOR1", "FINE1",
        "FLOAT1", "FOREIGNER1", "GUESS1", "HALLOWEEN1", "HOSPITAL1",
        "HURDLE/TRIP1", "KNIGHT1", "LOCK1", "LUNCH1", "MEAT1",
        "MECHANIC1", "MICROSCOPE1", "MOVIE1", "NOON1", "PARTY1",
        "PATIENT2", "RAZOR2", "RIVER1", "ROCKINGCHAIR1", "SHAVE1",
        "THEY1", "TWINS1",
    }
)

# ASL-LEX code C_02_054 is shared by WHATFOR1/2/3 in the source metadata.
CANONICAL_LABELS = {
    "WHATFOR1": "WHAT FOR",
    "WHATFOR2": "WHAT FOR",
    "WHATFOR3": "WHAT FOR",
}


def canonical_label(original_label: str) -> str:
    label = original_label.strip().upper()
    variant_marker = " (VARIANT "
    if label.endswith(")") and variant_marker in label:
        base, variant = label.rsplit(variant_marker, 1)
        variant_number = variant[:-1]
        if base and variant_number.isdigit():
            return f"{base} (variant {variant_number})"
    if label in CANONICAL_LABELS:
        return CANONICAL_LABELS[label]
    if label in DISTINCT_VARIANT_LABELS:
        base = label.rstrip("0123456789")
        variant = label[len(base):]
        return f"{base} (variant {variant})"
    return label


@dataclass(frozen=True)
class CanonicalPrediction:
    canonical_label: str
    confidence: float
    original_model_label: str
    class_index: int
    original_confidence: float


def aggregate_predictions(
    probabilities: torch.Tensor, vocab: list[str], top_k: int = 3
) -> list[CanonicalPrediction]:
    """Rank canonical signs by summed probability, retaining best source class."""
    if probabilities.ndim != 1 or probabilities.numel() != len(vocab):
        raise ValueError("probabilities must have one score for each vocabulary label")
    if top_k < 1:
        raise ValueError("top_k must be at least one")

    grouped: dict[str, list[tuple[int, float]]] = {}
    for class_index, score in enumerate(probabilities.tolist()):
        label = canonical_label(vocab[class_index])
        grouped.setdefault(label, []).append((class_index, score))

    predictions = []
    for label, members in grouped.items():
        class_index, original_confidence = max(members, key=lambda item: item[1])
        predictions.append(
            CanonicalPrediction(
                canonical_label=label,
                confidence=sum(score for _, score in members),
                original_model_label=vocab[class_index],
                class_index=class_index,
                original_confidence=original_confidence,
            )
        )
    return sorted(predictions, key=lambda item: item.confidence, reverse=True)[:top_k]
