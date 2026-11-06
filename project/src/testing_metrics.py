"""Metrics for independently reviewed Testing Mode attempts."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable

from label_display import canonical_label


def build_report(
    attempts: Iterable[Any],
    *,
    session: Any | None = None,
    minimum_sign_samples: int = 3,
    minimum_confusion_samples: int = 5,
) -> dict[str, Any]:
    """Build metrics from persisted Testing Mode attempts only.

    Correctness is evaluated on canonical labels. The only many-to-one mapping
    currently approved by metadata is WHATFOR1/2/3 -> WHAT FOR.
    """
    rows = [dict(attempt) for attempt in attempts]

    def prediction(row: dict[str, Any]) -> str:
        return canonical_label(
            str(row.get("canonical_label") or row.get("predicted_label") or "")
        )

    def normalize(value: Any) -> str:
        return canonical_label(str(value or ""))

    completed = [row for row in rows if row.get("status", "completed") == "completed"]
    valid = [row for row in completed if prediction(row)]
    verified = [
        row for row in valid
        if row.get("correctness_status") in {"correct", "incorrect"}
        and row.get("ground_truth_label")
    ]
    unverified = [
        row for row in valid
        if row.get("correctness_status", "unverified") == "unverified"
        or not row.get("ground_truth_label")
    ]

    target_comparable = [
        row for row in valid
        if row.get("target_label") and prediction(row)
    ]
    target_correct = sum(
        normalize(row["target_label"]) == prediction(row)
        for row in target_comparable
    )
    verified_correct = sum(
        normalize(row["ground_truth_label"]) == prediction(row)
        for row in verified
    )
    confidence_buckets: dict[str, list[bool]] = {
        "low (<50%)": [],
        "medium (50-<75%)": [],
        "high (>=75%)": [],
    }
    for row in verified:
        confidence = float(row.get("confidence") or 0.0)
        bucket = (
            "low (<50%)"
            if confidence < 0.5
            else "medium (50-<75%)"
            if confidence < 0.75
            else "high (>=75%)"
        )
        confidence_buckets[bucket].append(
            normalize(row["ground_truth_label"]) == prediction(row)
        )

    top_k_samples: dict[int, list[tuple[str, list[str]]]] = {
        3: [],
        5: [],
    }
    per_sign: dict[str, list[bool]] = defaultdict(list)
    confusion: Counter[tuple[str, str]] = Counter()
    for row in verified:
        truth = normalize(row["ground_truth_label"])
        predicted = prediction(row)
        per_sign[truth].append(truth == predicted)
        if truth != predicted:
            confusion[(truth, predicted)] += 1

        encoded = row.get("top_k_predictions") or "[]"
        try:
            candidates = json.loads(encoded) if isinstance(encoded, str) else encoded
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"Invalid top-K data for attempt {row.get('attempt_id')}"
            ) from exc
        if not isinstance(candidates, list):
            raise ValueError(
                f"Top-K data for attempt {row.get('attempt_id')} must be a list"
            )
        labels: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or not candidate.get("canonical_label"):
                raise ValueError(
                    f"Malformed top-K candidate for attempt {row.get('attempt_id')}"
                )
            label = normalize(candidate["canonical_label"])
            if label and label not in labels:
                labels.append(label)
        for k in top_k_samples:
            if len(labels) >= k:
                top_k_samples[k].append((truth, labels[:k]))

    per_sign_rows = [
        {
            "ground_truth": label,
            "samples": len(outcomes),
            "correct": sum(outcomes),
            "accuracy": sum(outcomes) / len(outcomes),
            "rankable": len(outcomes) >= minimum_sign_samples,
        }
        for label, outcomes in sorted(per_sign.items())
    ]
    rankable_signs = [row for row in per_sign_rows if row["rankable"]]
    best_signs = sorted(
        rankable_signs,
        key=lambda row: (-row["accuracy"], -row["samples"], row["ground_truth"]),
    )
    worst_signs = sorted(
        rankable_signs,
        key=lambda row: (row["accuracy"], -row["samples"], row["ground_truth"]),
    )

    matrix = None
    matrix_warning = None
    if len(verified) >= minimum_confusion_samples:
        labels = sorted(
            {
                normalize(row["ground_truth_label"])
                for row in verified
            }
            | {prediction(row) for row in verified}
        )
        counts = {truth: {guess: 0 for guess in labels} for truth in labels}
        for row in verified:
            counts[normalize(row["ground_truth_label"])][prediction(row)] += 1
        matrix = {"labels": labels, "counts": counts, "sample_count": len(verified)}
    else:
        matrix_warning = (
            f"Confusion matrix withheld: {len(verified)} verified valid samples; "
            f"at least {minimum_confusion_samples} are required."
        )

    statuses = Counter(str(row.get("status", "completed")) for row in rows)
    coverage_denominator = len(valid) + statuses["no_prediction"]
    answered_verified = [
        row for row in verified
        if row.get("status", "completed") == "completed"
    ]
    missing_prediction = sum(
        row.get("status", "completed") == "completed" and not prediction(row)
        for row in rows
    )
    no_demo = sum(not row.get("demo_video_id") for row in valid)
    reliability_notes = []
    if len(verified) < 30:
        reliability_notes.append(
            f"Small verified sample ({len(verified)}); results are descriptive, not a "
            "reliable estimate of generalization."
        )
    if valid and len(unverified) / len(valid) > 0.2:
        reliability_notes.append(
            f"{len(unverified)} of {len(valid)} valid attempts are unverified."
        )
    if no_demo:
        reliability_notes.append(
            f"{no_demo} valid attempts had no curated reference video."
        )
    reliability_notes.extend(
        [
            "No webcam recording of the performed sign is stored; retrospective "
            "verification requires an independent observer or external evidence.",
            "Signer IDs are optional and should be pseudonymous; repeated-user effects "
            "cannot be analyzed when IDs are omitted. This controlled prototype does "
            "not establish performance in other environments.",
        ]
    )

    return {
        "session": dict(session) if session is not None else None,
        "counts": {
            "total": len(rows),
            "completed": statuses["completed"],
            "valid": len(valid),
            "verified": len(verified),
            "unverified": len(unverified),
            "invalid": statuses["invalid"],
            "cancelled": statuses["cancelled"],
            "no_prediction": statuses["no_prediction"],
            "interrupted": statuses["interrupted"],
            "error": statuses["error"],
            "in_progress": statuses["in_progress"],
            "missing_prediction": missing_prediction,
        },
        "target_agreement": {
            "correct": target_correct,
            "total": len(target_comparable),
            "accuracy": (
                target_correct / len(target_comparable) if target_comparable else None
            ),
        },
        "verified_top1": {
            "correct": verified_correct,
            "total": len(verified),
            "accuracy": verified_correct / len(verified) if verified else None,
        },
        "coverage": {
            "total_attempts": len(rows),
            "eligible_attempts": coverage_denominator,
            "answered": len(valid),
            "rejected": statuses["no_prediction"],
            "coverage": (
                len(valid) / coverage_denominator if coverage_denominator else None
            ),
            "answered_verified_correct": sum(
                normalize(row["ground_truth_label"]) == prediction(row)
                for row in answered_verified
            ),
            "answered_verified_total": len(answered_verified),
            "answered_verified_accuracy": (
                sum(
                    normalize(row["ground_truth_label"]) == prediction(row)
                    for row in answered_verified
                )
                / len(answered_verified)
                if answered_verified else None
            ),
        },
        "confidence_analysis": {
            label: {
                "correct": sum(outcomes),
                "incorrect": len(outcomes) - sum(outcomes),
                "total": len(outcomes),
                "accuracy": sum(outcomes) / len(outcomes) if outcomes else None,
            }
            for label, outcomes in confidence_buckets.items()
        },
        "verified_top_k": {
            k: {
                "correct": sum(
                    truth in labels for truth, labels in values
                ),
                "total": len(values),
                "accuracy": (
                    sum(truth in labels for truth, labels in values) / len(values)
                    if values else None
                ),
            }
            for k, values in top_k_samples.items()
        },
        "per_sign": per_sign_rows,
        "best_signs": best_signs,
        "worst_signs": worst_signs,
        "common_confusions": [
            {"ground_truth": pair[0], "prediction": pair[1], "count": count}
            for pair, count in confusion.most_common(10)
        ],
        "confusion_matrix": matrix,
        "confusion_matrix_warning": matrix_warning,
        "minimum_sign_samples": minimum_sign_samples,
        "reliability_notes": reliability_notes,
        "label_policy": (
            "User-facing metrics compare canonical labels. WHATFOR1/2/3 are combined "
            "as WHAT FOR because source metadata identifies them as the same sign; "
            "their distinct checkpoint class indices remain stored. Other explicitly "
            "distinct suffix variants remain separate."
        ),
    }


def format_report(report: dict[str, Any]) -> str:
    """Render a concise, human-readable Testing Mode report."""
    session = report["session"] or {}
    counts = report["counts"]
    lines = [
        "TESTING MODE REPORT",
        f"Session: {session.get('session_id', 'all sessions')}",
        f"Status: {session.get('status', 'all')}",
        f"Started: {session.get('started_at', 'not recorded')}",
        f"Model: {session.get('model_version', 'mixed / not recorded')}",
        f"Signer: {session.get('signer_id') or 'not recorded'}",
        f"Attempts: {counts['total']} total · {counts['completed']} completed · "
        f"{counts['valid']} valid · {counts['verified']} verified",
        "Data quality: "
        f"{counts['unverified']} unverified · {counts['invalid']} invalid · "
        f"{counts['cancelled']} cancelled · {counts['interrupted']} interrupted · "
        f"{counts['error']} errors · {counts['no_prediction']} no prediction · "
        f"{counts['missing_prediction']} missing prediction · "
        f"{counts['in_progress']} in progress",
    ]
    target = report["target_agreement"]
    lines.append(
        "Target Agreement: "
        + _format_rate(target["accuracy"], target["correct"], target["total"])
        + " (prediction vs requested target; not verified accuracy)"
    )
    top1 = report["verified_top1"]
    lines.append(
        "Verified Top-1 Accuracy: "
        + _format_rate(top1["accuracy"], top1["correct"], top1["total"])
    )
    coverage = report["coverage"]
    lines.append(
        "Coverage: "
        + _format_rate(
            coverage["coverage"], coverage["answered"], coverage["eligible_attempts"]
        )
        + f" · {coverage['answered']} answered · {coverage['rejected']} rejected"
    )
    lines.append(
        "Verified accuracy among answered: "
        + _format_rate(
            coverage["answered_verified_accuracy"],
            coverage["answered_verified_correct"],
            coverage["answered_verified_total"],
        )
    )
    lines.extend(["", "VERIFIED ACCURACY BY CONFIDENCE"])
    for label, result in report["confidence_analysis"].items():
        lines.append(
            f"{label}: "
            + _format_rate(result["accuracy"], result["correct"], result["total"])
        )
    for k, result in report["verified_top_k"].items():
        if result["total"] == 0:
            continue
        lines.append(
            f"Verified Top-{k} Accuracy: "
            + _format_rate(result["accuracy"], result["correct"], result["total"])
        )
    lines.extend(["", "PER-SIGN VERIFIED RESULTS"])
    if not report["per_sign"]:
        lines.append("No valid verified samples.")
    for row in report["per_sign"]:
        rate = _format_rate(row["accuracy"], row["correct"], row["samples"])
        warning = "" if row["rankable"] else " · below ranking minimum"
        lines.append(f"{row['ground_truth']}: {rate}{warning}")
    lines.append(
        f"Best signs (at least {report['minimum_sign_samples']} verified samples):"
    )
    lines.extend(
        f"{row['ground_truth']}: "
        f"{_format_rate(row['accuracy'], row['correct'], row['samples'])}"
        for row in report["best_signs"][:5]
    )
    if not report["best_signs"]:
        lines.append("Not enough verified samples to rank signs.")
    lines.append(
        f"Worst signs (at least {report['minimum_sign_samples']} verified samples):"
    )
    lines.extend(
        f"{row['ground_truth']}: "
        f"{_format_rate(row['accuracy'], row['correct'], row['samples'])}"
        for row in report["worst_signs"][:5]
    )
    if not report["worst_signs"]:
        lines.append("Not enough verified samples to rank signs.")
    if report["confusion_matrix"] is None:
        lines.extend(["", report["confusion_matrix_warning"]])
    else:
        lines.extend(["", "CONFUSION MATRIX — Testing Mode verified attempts"])
        matrix = report["confusion_matrix"]
        lines.append("Rows = ground truth; columns = prediction")
        lines.append("Ground truth | " + " | ".join(matrix["labels"]))
        for truth in matrix["labels"]:
            lines.append(
                f"{truth} | "
                + " | ".join(str(matrix["counts"][truth][guess]) for guess in matrix["labels"])
            )
    if report["common_confusions"]:
        lines.extend(["", "COMMON MISCLASSIFICATIONS"])
        lines.extend(
            f"{row['ground_truth']} -> {row['prediction']}: {row['count']}"
            for row in report["common_confusions"]
        )
    lines.extend(["", "LABEL POLICY", report["label_policy"], "", "RELIABILITY"])
    lines.extend(f"- {note}" for note in report["reliability_notes"])
    return "\n".join(lines)


def _format_rate(accuracy: float | None, correct: int, total: int) -> str:
    if accuracy is None or total == 0:
        return f"not available (0 / {total})"
    return f"{accuracy:.1%} ({correct} / {total} attempts)"
