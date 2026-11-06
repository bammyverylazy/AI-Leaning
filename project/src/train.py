"""ฝึก baseline GRU/LSTM แล้วประเมินบน test (ผู้แสดงที่ไม่เคยเห็น)

ใช้งาน:
  python src/train.py --kind gru --epochs 60 --seed 0
  python src/train.py --kind lstm --epochs 60 --seed 0
ผลลัพธ์: models/<kind>_candidate.pt, reports/<kind>_candidate_*.{json,png,csv}
"""
import argparse
import csv
import hashlib
import json
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

import config as C
from dataset import FEATURE_DIM, SignDataset
from model import SeqClassifier
from model_vocabulary import build_class_mapping, checkpoint_vocabulary, validate_vocabulary


class FeatureSplitSummary(TypedDict):
    samples: int
    counts: Counter[int]
    participants: set[str]


def validate_feature_archive(
    split: str,
    archive_path: Path,
    vocab: list[str],
) -> FeatureSplitSummary:
    """Validate dimensions, class indices, and legacy label mappings before training."""
    subset_path = C.DATA / f"subset_{split}.csv"
    if not subset_path.is_file():
        raise FileNotFoundError(f"Required source subset is missing: {subset_path}")
    subset = pd.read_csv(subset_path)
    if not {"video", "gloss"}.issubset(subset.columns):
        raise ValueError(f"{subset_path} must contain video and gloss columns.")
    if subset["video"].astype(str).duplicated().any():
        raise ValueError(f"{subset_path} contains duplicate video IDs.")
    label_by_video = dict(
        zip(subset["video"].astype(str), subset["gloss"].astype(str), strict=True)
    )
    index_by_label = {label: index for index, label in enumerate(vocab)}
    unknown = sorted(set(label_by_video.values()) - set(index_by_label))
    if unknown:
        raise ValueError(
            f"{subset_path} contains labels not present in the training vocabulary: "
            + ", ".join(unknown)
        )

    with np.load(archive_path, allow_pickle=True) as data:
        required = {"pos", "present", "y", "video", "participant"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{archive_path} is missing arrays: {', '.join(missing)}")
        positions = data["pos"]
        presence = data["present"]
        labels = np.asarray(data["y"], dtype=np.int64)
        videos = np.asarray(data["video"]).astype(str)
        participants = np.asarray(data["participant"]).astype(str)
        count = len(labels)
        expected_shapes = {
            "pos": (count, C.SEQ_LEN, C.N_POINTS, 3),
            "present": (count, C.SEQ_LEN, 2),
            "y": (count,),
            "video": (count,),
            "participant": (count,),
        }
        actual_shapes = {
            "pos": positions.shape,
            "present": presence.shape,
            "y": labels.shape,
            "video": videos.shape,
            "participant": participants.shape,
        }
        if actual_shapes != expected_shapes:
            raise ValueError(
                f"{archive_path} has incompatible array shapes: {actual_shapes}"
            )
        if not np.isfinite(positions).all() or not np.isfinite(presence).all():
            raise ValueError(f"{archive_path} contains non-finite feature inputs.")
        if labels.size and (labels.min() < 0 or labels.max() >= len(vocab)):
            raise ValueError(f"{archive_path} contains out-of-range class indices.")
        if len(set(videos)) != len(videos):
            raise ValueError(f"{archive_path} contains duplicate video IDs.")
        unknown_videos = sorted(set(videos) - set(label_by_video))
        if unknown_videos:
            raise ValueError(
                f"{archive_path} includes videos absent from {subset_path}: "
                + ", ".join(unknown_videos[:10])
            )
        expected_labels = np.asarray(
            [index_by_label[label_by_video[video]] for video in videos],
            dtype=np.int64,
        )
        if not np.array_equal(labels, expected_labels):
            raise ValueError(
                f"{archive_path} class indices do not match {subset_path} and "
                "data/vocab.txt; regenerate features before training."
            )
        if "vocab" in data.files:
            archived_vocab = [str(label) for label in data["vocab"].tolist()]
            if archived_vocab != vocab:
                raise ValueError(
                    f"{archive_path} was generated with a different vocabulary order."
                )
        if "class_mapping_json" in data.files:
            archived_mapping = json.loads(str(data["class_mapping_json"].item()))
            if archived_mapping != build_class_mapping(vocab):
                raise ValueError(
                    f"{archive_path} class mapping differs from data/vocab.txt."
                )

    return {
        "samples": count,
        "counts": Counter(map(int, labels)),
        "participants": set(participants),
    }


def validate_signer_splits(split_data: dict[str, FeatureSplitSummary]) -> None:
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(split_data[left]["participants"]) & set(
            split_data[right]["participants"]
        )
        if overlap:
            raise ValueError(
                f"Signer leakage between {left} and {right}: "
                + ", ".join(sorted(overlap))
            )


def load_self_recorded_feature_archive(
    archive_path: Path,
    split: str,
    vocab: list[str],
) -> dict[str, np.ndarray]:
    """Load only explicitly verified, quality-passed samples for their assigned split."""
    required = {
        "pos",
        "present",
        "y",
        "participant",
        "video",
        "source",
        "sample_id",
        "signer_id",
        "session_id",
        "target_label",
        "ground_truth_label",
        "recorded_at",
        "dominant_hand",
        "camera_source",
        "camera_index",
        "camera_angle",
        "distance",
        "lighting",
        "signing_speed",
        "recording_status",
        "verification_status",
        "quality_status",
        "dataset_split",
        "sequence_sha256",
        "vocab",
        "class_mapping_json",
    }
    with np.load(archive_path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                f"{archive_path} is missing required provenance arrays: "
                + ", ".join(missing)
            )
        data = {name: np.array(archive[name], copy=True) for name in required}
    count = len(data["y"])
    if data["y"].ndim != 1 or not np.issubdtype(data["y"].dtype, np.integer):
        raise ValueError(f"{archive_path} has invalid class-index dimensions or type.")
    row_fields = required - {
        "pos",
        "present",
        "vocab",
        "class_mapping_json",
        "camera_index",
    }
    if data["pos"].shape != (count, C.SEQ_LEN, C.N_POINTS, 3):
        raise ValueError(f"{archive_path} has invalid position dimensions.")
    if data["present"].shape != (count, C.SEQ_LEN, 2):
        raise ValueError(f"{archive_path} has invalid hand-presence dimensions.")
    if data["camera_index"].shape != (count,) or not np.issubdtype(
        data["camera_index"].dtype, np.integer
    ):
        raise ValueError(f"{archive_path} has invalid camera_index dimensions.")
    for field in row_fields:
        if data[field].shape != (count,):
            raise ValueError(f"{archive_path} has invalid {field} dimensions.")
        data[field] = data[field].astype(str)
    if not np.isfinite(data["pos"]).all() or not np.isfinite(data["present"]).all():
        raise ValueError(f"{archive_path} contains non-finite inputs.")
    if not np.isin(data["present"], (0, 1)).all():
        raise ValueError(f"{archive_path} has invalid hand-presence flags.")
    if [str(label) for label in data["vocab"].tolist()] != vocab:
        raise ValueError(f"{archive_path} vocabulary order differs from data/vocab.txt.")
    mapping = json.loads(str(data["class_mapping_json"].item()))
    if mapping != build_class_mapping(vocab):
        raise ValueError(f"{archive_path} has an incompatible class mapping.")
    assigned_split = "val" if split == "validation" else split
    expected_values = {
        "source": "self_recorded",
        "recording_status": "VERIFIED",
        "verification_status": "VERIFIED",
        "quality_status": "PASSED",
        "dataset_split": assigned_split,
    }
    for field, expected in expected_values.items():
        if not np.all(data[field] == expected):
            raise ValueError(
                f"{archive_path} contains samples without verified {field}={expected}."
            )
    for field in (
        "signer_id",
        "session_id",
        "sample_id",
        "sequence_sha256",
        "target_label",
        "recorded_at",
        "camera_source",
    ):
        if np.any(np.char.str_len(data[field]) == 0):
            raise ValueError(f"{archive_path} contains an empty {field}.")
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower())
        for value in data["sequence_sha256"]
    ):
        raise ValueError(f"{archive_path} contains an invalid sequence SHA-256.")
    if not np.isin(data["dominant_hand"], ("RIGHT", "LEFT", "UNKNOWN")).all():
        raise ValueError(f"{archive_path} contains an invalid dominant-hand value.")
    if np.any(data["camera_index"] < 0):
        raise ValueError(f"{archive_path} contains an invalid camera index.")
    if len(set(data["sample_id"])) != count:
        raise ValueError(f"{archive_path} contains duplicate sample IDs.")
    if len(set(data["sequence_sha256"])) != count:
        raise ValueError(f"{archive_path} contains duplicate sequences.")
    if not np.array_equal(
        data["participant"],
        np.asarray([f"self_recorded:{value}" for value in data["signer_id"]]),
    ):
        raise ValueError(f"{archive_path} participant provenance is inconsistent.")
    if not np.array_equal(
        data["video"],
        np.asarray([f"self_recorded:{value}" for value in data["sample_id"]]),
    ):
        raise ValueError(f"{archive_path} sample provenance is inconsistent.")
    index_by_label = {label: index for index, label in enumerate(vocab)}
    unknown_labels = sorted(set(data["ground_truth_label"]) - set(index_by_label))
    if unknown_labels:
        raise ValueError(
            f"{archive_path} contains labels not in the model vocabulary: "
            + ", ".join(unknown_labels)
        )
    unknown_targets = sorted(set(data["target_label"]) - set(index_by_label))
    if unknown_targets:
        raise ValueError(
            f"{archive_path} contains target labels not in the model vocabulary: "
            + ", ".join(unknown_targets)
        )
    expected_y = np.asarray(
        [index_by_label[label] for label in data["ground_truth_label"]],
        dtype=np.int64,
    )
    if not np.array_equal(data["y"].astype(np.int64), expected_y):
        raise ValueError(f"{archive_path} class indices do not match verified labels.")
    data["y"] = expected_y
    return data


def append_self_recorded_split(
    dataset: SignDataset,
    summary: FeatureSplitSummary,
    data: dict[str, np.ndarray],
) -> None:
    count = len(data["y"])
    if count == 0:
        return
    if set(dataset.video.astype(str)) & set(data["video"]):
        raise ValueError("A self-recorded sample ID is already in the source dataset.")
    dataset.pos = np.concatenate((dataset.pos, data["pos"]), axis=0)
    dataset.present = np.concatenate((dataset.present, data["present"]), axis=0)
    dataset.y = np.concatenate((dataset.y, data["y"]), axis=0)
    dataset.participant = np.concatenate(
        (dataset.participant.astype(str), data["participant"]), axis=0
    )
    dataset.video = np.concatenate(
        (dataset.video.astype(str), data["video"]), axis=0
    )
    summary["samples"] += count
    summary["counts"].update(map(int, data["y"]))
    summary["participants"].update(map(str, data["participant"]))


def validate_unique_self_recorded_splits(
    split_data: dict[str, dict[str, np.ndarray]],
) -> None:
    sample_ids: set[str] = set()
    sequence_hashes: set[str] = set()
    for data in split_data.values():
        ids = set(map(str, data["sample_id"]))
        hashes = set(map(str, data["sequence_sha256"]))
        if sample_ids & ids or sequence_hashes & hashes:
            raise ValueError(
                "Self-recorded feature archives duplicate a sample or sequence "
                "across dataset splits."
            )
        sample_ids.update(ids)
        sequence_hashes.update(hashes)


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_candidate_checkpoint_path(path: Path) -> None:
    protected_baseline = (C.MODELS / "gru_seed3.pt").resolve()
    if path.resolve() == protected_baseline:
        raise ValueError(
            "The baseline checkpoint gru_seed3.pt is protected; choose a separate "
            "candidate checkpoint path."
        )


def evaluate(model, loader, device):
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for x, y in loader:
            logits_all.append(model(x.to(device)).cpu())
            y_all.append(y)
    return torch.cat(logits_all), torch.cat(y_all)


def topk_acc(logits, y, k):
    k = min(k, logits.shape[1])
    return (logits.topk(k, dim=1).indices == y[:, None]).any(dim=1).float().mean().item()


def classification_summary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocabulary: list[str],
) -> dict[str, Any]:
    predictions = logits.argmax(1).numpy()
    labels = targets.numpy()
    per_class = []
    class_counts = []
    for index, label in enumerate(vocabulary):
        selected = labels == index
        true_positive = int(((predictions == index) & selected).sum())
        false_positive = int(((predictions == index) & ~selected).sum())
        false_negative = int(((predictions != index) & selected).sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / (true_positive + false_negative) if selected.any() else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        class_counts.append({"label": label, "samples": int(selected.sum())})
        per_class.append(
            {
                "class_index": index,
                "label": label,
                "test_samples": int(selected.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "top1": float(accuracy_score(labels, predictions)),
        "top5": topk_acc(logits, targets, 5),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "per_class": per_class,
        "class_counts": class_counts,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=list(range(len(vocabulary)))
        ).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["gru", "lstm"], default="gru")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="checkpoint path; defaults to a separate candidate checkpoint",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="back up an existing output checkpoint before replacing it",
    )
    ap.add_argument(
        "--self-recorded-features-dir",
        type=Path,
        default=None,
        help=(
            "explicit opt-in directory created by manage_recordings.py "
            "prepare-features; verified data is combined with the matching splits"
        ),
    )
    ap.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=C.MODELS / "gru_seed3.pt",
        help="evaluate this compatible baseline on the same held-out test protocol",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = validate_vocabulary(
        [
            label.strip()
            for label in (C.DATA / "vocab.txt").read_text(encoding="utf-8").splitlines()
            if label.strip()
        ]
    )
    n_classes = len(vocab)

    split_data = {
        split: validate_feature_archive(
            split, C.FEATURES / f"{split}.npz", vocab
        )
        for split in ("train", "val", "test")
    }
    ckpt = args.output or C.MODELS / f"{args.kind}_candidate.pt"
    if not ckpt.is_absolute():
        ckpt = C.ROOT / ckpt
    ensure_candidate_checkpoint_path(ckpt)

    train_ds = SignDataset(C.FEATURES / "train.npz", augment_data=True)
    val_ds = SignDataset(C.FEATURES / "val.npz")
    test_ds = SignDataset(C.FEATURES / "test.npz")
    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    original_class_counts = {
        split: {
            vocab[index]: int(summary["counts"].get(index, 0))
            for index in range(n_classes)
        }
        for split, summary in split_data.items()
    }
    self_recorded_sample_ids: dict[str, list[str]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    self_recorded_splits: dict[str, dict[str, np.ndarray]] = {
        split: {"y": np.empty(0, dtype=np.int64)}
        for split in ("train", "val", "test")
    }
    dataset_hashes = {
        split: archive_sha256(C.FEATURES / f"{split}.npz")
        for split in ("train", "val", "test")
    }
    if args.self_recorded_features_dir is not None:
        feature_dir = args.self_recorded_features_dir
        if not feature_dir.is_absolute():
            feature_dir = C.ROOT / feature_dir
        feature_paths = {
            "train": feature_dir / "train.npz",
            "val": feature_dir / "val.npz",
            "test": feature_dir / "test.npz",
        }
        for split, path in feature_paths.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"Self-recorded {split} archive is missing: {path}. "
                    "Create it with manage_recordings.py prepare-features."
                )
            extra = load_self_recorded_feature_archive(
                path, split, vocab
            )
            self_recorded_splits[split] = extra
            dataset_hashes[f"self_recorded_{split}"] = archive_sha256(path)
        validate_unique_self_recorded_splits(self_recorded_splits)
        for split, extra in self_recorded_splits.items():
            append_self_recorded_split(datasets[split], split_data[split], extra)
            self_recorded_sample_ids[split] = list(map(str, extra["sample_id"]))
        if not self_recorded_sample_ids["train"]:
            raise ValueError(
                "The self-recorded feature directory contains no verified training "
                "samples. No candidate training was started."
            )
    validate_signer_splits(split_data)
    for split, result in split_data.items():
        counts = result["counts"]
        missing_classes = [
            vocab[index] for index in range(n_classes) if not counts.get(index, 0)
        ]
        if missing_classes:
            raise ValueError(
                f"{split} features are missing classes: "
                + ", ".join(missing_classes)
            )
    final_class_counts = {
        split: {
            vocab[index]: int(summary["counts"].get(index, 0))
            for index in range(n_classes)
        }
        for split, summary in split_data.items()
    }
    self_recorded_class_counts = {
        split: {
            vocab[index]: int(
                np.count_nonzero(self_recorded_splits[split]["y"] == index)
            )
            for index in range(n_classes)
        }
        for split in ("train", "val", "test")
    }
    self_recorded_classes_without_training_samples = [
        label
        for label in vocab
        if self_recorded_class_counts["train"][label] == 0
    ]
    dataset_version = hashlib.sha256(
        json.dumps(
            {"archive_hashes": dataset_hashes, "vocabulary": vocab},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if ckpt.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Checkpoint already exists: {ckpt}. Choose another --output or "
                "use --overwrite to make a backup before replacing it."
            )
        backup = ckpt.with_name(
            f"{ckpt.stem}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            f".bak{ckpt.suffix}"
        )
        shutil.copy2(ckpt, backup)
        print(f"Existing checkpoint backed up to {backup}")
    # num_workers=0 ปลอดภัยบน Windows (ไม่ต้องกังวลเรื่อง spawn)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=len(train_ds) > args.batch)
    val_dl = DataLoader(val_ds, batch_size=128)
    test_dl = DataLoader(test_ds, batch_size=128)
    print(
        f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}  "
        f"classes {n_classes}  device {device}"
    )
    print(
        "Class counts before integration: "
        + json.dumps(original_class_counts, sort_keys=True)
    )
    print(
        "Class counts after integration: "
        + json.dumps(final_class_counts, sort_keys=True)
    )

    model = SeqClassifier(FEATURE_DIM, n_classes, args.hidden, 2, args.dropout, args.kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    C.MODELS.mkdir(exist_ok=True)
    C.REPORTS.mkdir(exist_ok=True)
    tag = ckpt.stem
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best, bad = -1.0, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()
        vl, vy = evaluate(model, val_dl, device)
        vloss = nn.functional.cross_entropy(vl, vy).item()
        vacc = (vl.argmax(1) == vy).float().mean().item()
        hist["train_loss"].append(tot / n)
        hist["val_loss"].append(vloss)
        hist["val_acc"].append(vacc)
        print(f"ep {ep:3d}  train_loss {tot / n:.3f}  val_loss {vloss:.3f}  val_acc {vacc:.3f}  ({time.time() - t0:.1f}s)")
        if vacc > best:
            best, bad = vacc, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "vocab": vocab,
                    "class_mapping": build_class_mapping(vocab),
                    "feature_dim": FEATURE_DIM,
                    "dataset_version": dataset_version,
                    "dataset_hashes": dataset_hashes,
                    "class_counts_before_integration": original_class_counts,
                    "class_counts_after_integration": final_class_counts,
                    "self_recorded_class_counts": self_recorded_class_counts,
                    "self_recorded_classes_without_training_samples": (
                        self_recorded_classes_without_training_samples
                    ),
                    "self_recorded_sample_ids": self_recorded_sample_ids,
                },
                ckpt,
            )
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    # ---- ประเมินด้วย checkpoint ที่ดีที่สุดบน validation ----
    best_checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    if checkpoint_vocabulary(best_checkpoint) != vocab:
        raise ValueError("Saved checkpoint vocabulary changed during training.")
    model.load_state_dict(best_checkpoint["model"])
    tl, ty = evaluate(model, test_dl, device)
    vl, vy = evaluate(model, val_dl, device)
    pred = tl.argmax(1).numpy()
    y_np = ty.numpy()
    candidate_summary = classification_summary(tl, ty, vocab)
    baseline_path = args.baseline_checkpoint
    if not baseline_path.is_absolute():
        baseline_path = C.ROOT / baseline_path
    baseline_summary: dict[str, Any] = {
        "available": False,
        "reason": f"Baseline checkpoint not found: {baseline_path}",
    }
    if baseline_path.is_file():
        baseline_checkpoint = torch.load(
            baseline_path, map_location=device, weights_only=False
        )
        if checkpoint_vocabulary(baseline_checkpoint) != vocab:
            raise ValueError(
                "Baseline checkpoint vocabulary differs from the candidate evaluation."
            )
        if int(baseline_checkpoint["feature_dim"]) != FEATURE_DIM:
            raise ValueError(
                "Baseline checkpoint feature dimension differs from the evaluation data."
            )
        baseline_args = baseline_checkpoint["args"]
        baseline_model = SeqClassifier(
            baseline_checkpoint["feature_dim"],
            len(vocab),
            baseline_args["hidden"],
            2,
            baseline_args["dropout"],
            baseline_args["kind"],
        ).to(device)
        baseline_model.load_state_dict(baseline_checkpoint["model"])
        baseline_logits, baseline_targets = evaluate(
            baseline_model, test_dl, device
        )
        if not torch.equal(baseline_targets, ty):
            raise ValueError(
                "Baseline and candidate were not evaluated on identical test labels."
            )
        baseline_summary = {
            "available": True,
            "checkpoint": str(baseline_path.resolve()),
            "sha256": archive_sha256(baseline_path),
            **classification_summary(baseline_logits, baseline_targets, vocab),
        }

    metrics: dict[str, object] = {
        **candidate_summary,
        "best_val_acc": best,
        "dataset_version": dataset_version,
        "dataset_hashes": dataset_hashes,
        "class_counts_before_integration": original_class_counts,
        "class_counts_after_integration": final_class_counts,
        "self_recorded_class_counts": self_recorded_class_counts,
        "self_recorded_classes_without_training_samples": (
            self_recorded_classes_without_training_samples
        ),
        "self_recorded_sample_ids": self_recorded_sample_ids,
        "evaluation": {
            "split": "test",
            "candidate_checkpoint": str(ckpt.resolve()),
            "baseline": baseline_summary,
            "same_samples_and_labels": bool(
                baseline_summary.get("available", False)
            ),
        },
    }

    # เกณฑ์ความมั่นใจสำหรับตอบ "ไม่แน่ใจ" เลือกจาก validation (ไม่ใช่ test)
    vconf = torch.softmax(vl, 1).max(1).values.numpy()
    vok = (vl.argmax(1) == vy).numpy()
    thr = 0.0
    for t in np.linspace(0.05, 0.95, 19):
        keep = vconf >= t
        if keep.sum() >= 0.5 * len(vconf) and vok[keep].mean() >= 0.9:
            thr = float(t)
            break
    tconf = torch.softmax(tl, 1).max(1).values.numpy()
    keep = tconf >= thr
    metrics.update(
        {
            "conf_threshold": thr,
            "test_coverage_at_thr": float(keep.mean()),
            "test_acc_on_kept": float((pred[keep] == y_np[keep]).mean()) if keep.any() else None,
        }
    )
    (C.REPORTS / f"{tag}_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(json.dumps(metrics, indent=2))
    per_class = candidate_summary["per_class"]
    with (C.REPORTS / f"{tag}_per_class.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=per_class[0].keys())
        writer.writeheader()
        writer.writerows(per_class)

    # ---- accuracy แยกตามผู้แสดง ----
    parts = test_ds.participant
    lines = ["participant,n,acc"]
    for p in sorted(set(parts)):
        m = parts == p
        lines.append(f"{p},{int(m.sum())},{(pred[m] == y_np[m]).mean():.3f}")
    (C.REPORTS / f"{tag}_per_signer.csv").write_text("\n".join(lines))

    # ---- คู่คำที่สับสนกันมากที่สุด ----
    cm = confusion_matrix(y_np, pred, labels=list(range(n_classes)))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], vocab[i], vocab[j]) for i in range(n_classes) for j in range(n_classes) if off[i, j] > 0), reverse=True)[:20]
    (C.REPORTS / f"{tag}_top_confusions.csv").write_text("count,true,predicted\n" + "\n".join(f"{c},{a},{b}" for c, a, b in pairs))

    # ---- กราฟ ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(hist["val_acc"])
    ax[1].set_title("Validation accuracy")
    fig.tight_layout()
    fig.savefig(C.REPORTS / f"{tag}_curves.png", dpi=120)

    size = max(8, n_classes * 0.18)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(C.REPORTS / f"{tag}_confusion.png", dpi=120)
    print("reports saved in", C.REPORTS)


if __name__ == "__main__":
    main()
