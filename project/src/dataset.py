"""เตรียมข้อมูล: เติมเฟรมที่หาย, normalize, resample, augmentation, Dataset ของ PyTorch

สร้าง data/features/{train,val,test}.npz:
  pos [N, T, P, 3], present [N, T, 2], y [N], participant [N], video [N]

ใช้งาน:
  python src/dataset.py --min-hand-ratio 0.2
"""
import argparse
import csv
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config as C
from model_vocabulary import build_class_mapping, validate_vocabulary


# ---------- ฟังก์ชันพื้นฐาน ----------
def sample_at(x, t):
    """สุ่มค่าตามแกนเวลาแบบ linear: x [T, ...], t [n] (หน่วย = index เฟรม)"""
    T = x.shape[0]
    if T == 1:
        return np.repeat(x, len(t), axis=0)
    t = np.clip(t, 0, T - 1)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (t - lo).reshape(-1, *([1] * (x.ndim - 1)))
    return x[lo] * (1 - w) + x[hi] * w


def interpolate_nans(x):
    """เติม NaN ตามแกนเวลาของแต่ละพิกัด (ขอบใช้ค่าใกล้สุด); ที่หายทั้งคลิปยังเป็น NaN"""
    T = x.shape[0]
    flat = x.reshape(T, -1)
    ar = np.arange(T)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        valid = ~np.isnan(col)
        if valid.all() or not valid.any():
            continue
        col[~valid] = np.interp(ar[~valid], ar[valid], col[valid])
    return flat.reshape(x.shape)


def preprocess_sequence(raw, seq_len=C.SEQ_LEN, trim_margin=3):
    """raw [T, P, 3] (มี NaN) -> pos [seq_len, P, 3], present [seq_len, 2]
    ใช้ฟังก์ชันเดียวกันทั้งตอนฝึกและตอนเดโมเว็บแคม"""
    raw = raw.astype(np.float32).copy()
    present = np.stack(
        [~np.isnan(raw[:, C.LH, 0]).any(axis=1), ~np.isnan(raw[:, C.RH, 0]).any(axis=1)], axis=1
    )

    # ตัดช่วงต้น/ท้ายที่ไม่มีมือ (ท่าพักแขน) เหลือ margin เล็กน้อย
    idx = np.where(present.any(axis=1))[0]
    if len(idx) >= 2:
        lo, hi = max(0, idx[0] - trim_margin), min(len(raw), idx[-1] + trim_margin + 1)
        raw, present = raw[lo:hi], present[lo:hi]

    raw = interpolate_nans(raw)

    # normalize: ศูนย์กลาง = กึ่งกลางไหล่, สเกล = ความกว้างไหล่ (ค่ากลางตลอดคลิป)
    sh = raw[:, [C.L_SH, C.R_SH], :]
    if np.isnan(sh).all():
        center, scale = np.zeros(3, np.float32), 1.0
    else:
        center = np.nanmedian(np.nanmean(sh, axis=1), axis=0)
        scale = float(np.nanmedian(np.linalg.norm(sh[:, 0, :2] - sh[:, 1, :2], axis=1)))
        if not np.isfinite(scale) or scale < 1e-3:
            scale = 1.0
    pos = (raw - center) / scale

    t = np.linspace(0, len(pos) - 1, seq_len)
    pos = sample_at(pos, t)
    present = (sample_at(present.astype(np.float32), t) > 0.5).astype(np.float32)
    return np.nan_to_num(pos, nan=0.0).astype(np.float32), present


def build_features(pos, present):
    """pos [T,P,3], present [T,2] -> [T, F] (ตำแหน่ง + ความเร็ว + flag มือ)"""
    T = pos.shape[0]
    flat = pos.reshape(T, -1)
    vel = np.zeros_like(flat)
    vel[1:] = flat[1:] - flat[:-1]
    return np.concatenate([flat, vel, present], axis=1).astype(np.float32)


FEATURE_DIM = C.N_POINTS * 3 * 2 + 2


# ---------- Augmentation (ใช้เฉพาะชุดฝึก) ----------
def time_warp(pos, present, strength=0.3, knots=5):
    T = pos.shape[0]
    speed = np.interp(np.arange(T), np.linspace(0, T - 1, knots), np.random.uniform(1 - strength, 1 + strength, knots))
    t = np.cumsum(speed)
    t = (t - t[0]) / (t[-1] - t[0]) * (T - 1)
    return sample_at(pos, t), (sample_at(present, t) > 0.5).astype(np.float32)


def augment(pos, present):
    pos, present = pos.copy(), present.copy()
    zl = np.all(pos[:, C.LH] == 0, axis=(1, 2))  # มือที่ไม่เคยตรวจเจอ (เป็น 0) ให้คงเป็น 0
    zr = np.all(pos[:, C.RH] == 0, axis=(1, 2))
    pos, present = time_warp(pos, present)
    ang, s = np.deg2rad(np.random.uniform(-10, 10)), np.random.uniform(0.9, 1.1)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32) * s
    pos[..., :2] = pos[..., :2] @ R.T + np.random.uniform(-0.08, 0.08, size=2).astype(np.float32)
    pos[..., 2] *= s
    pos += np.random.normal(0, 0.005, pos.shape).astype(np.float32)
    # time_warp ทำให้ index เฟรมเปลี่ยน จึงสุ่มค่า zl/zr ใหม่ตามแกนเวลาใหม่ไม่ได้โดยตรง
    # ใช้การตัดสินจากความเป็นศูนย์ทั้งคลิปแทน
    if zl.all():
        pos[:, C.LH] = 0
    if zr.all():
        pos[:, C.RH] = 0
    return pos.astype(np.float32), present


class SignDataset(Dataset):
    def __init__(self, npz_path, augment_data=False):
        d = np.load(npz_path, allow_pickle=True)
        self.pos, self.present, self.y = d["pos"], d["present"], d["y"].astype(np.int64)
        self.participant, self.video = d["participant"], d["video"]
        self.augment_data = augment_data

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        pos, pres = self.pos[i], self.present[i]
        if self.augment_data:
            pos, pres = augment(pos, pres)
        return torch.from_numpy(build_features(pos, pres)), int(self.y[i])


# ---------- สร้างไฟล์ features ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hand-ratio", type=float, default=0.2, help="คัดวิดีโอที่ตรวจเจอมือน้อยกว่านี้ออก")
    args = ap.parse_args()

    vocab = validate_vocabulary(
        [
            label.strip()
            for label in (C.DATA / "vocab.txt").read_text(encoding="utf-8").splitlines()
            if label.strip()
        ]
    )
    g2i = {g: i for i, g in enumerate(vocab)}
    C.FEATURES.mkdir(parents=True, exist_ok=True)
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    report_rows = []

    for split in ["train", "val", "test"]:
        df = pd.read_csv(C.DATA / f"subset_{split}.csv")
        unknown_labels = sorted(set(df["gloss"].astype(str)) - set(g2i))
        if unknown_labels:
            raise ValueError(
                f"{split} subset contains labels missing from data/vocab.txt: "
                + ", ".join(unknown_labels)
            )
        P, PR, Y, PA, VI = [], [], [], [], []
        for r in df.itertuples():
            f = C.LANDMARKS / (r.video.rsplit(".", 1)[0] + ".npy")
            if not f.exists():
                report_rows.append(
                    {
                        "split": split,
                        "video": r.video,
                        "label": r.gloss,
                        "status": "missing_landmarks",
                        "frames": 0,
                        "hand_ratio": "",
                        "error": "Landmark file does not exist.",
                    }
                )
                continue
            try:
                raw = np.load(f)
                if raw.ndim != 3 or raw.shape[1:] != (C.N_POINTS, 3):
                    raise ValueError(
                        f"Expected [frames, {C.N_POINTS}, 3], got {raw.shape}."
                    )
                if len(raw) == 0:
                    raise ValueError("Landmark sequence is empty.")
                hand = (
                    ~np.isnan(raw[:, C.LH, 0]).any(axis=1)
                    | ~np.isnan(raw[:, C.RH, 0]).any(axis=1)
                )
                hand_ratio = float(hand.mean())
                if hand_ratio < args.min_hand_ratio:
                    report_rows.append(
                        {
                            "split": split,
                            "video": r.video,
                            "label": r.gloss,
                            "status": "low_hand_ratio",
                            "frames": len(raw),
                            "hand_ratio": hand_ratio,
                            "error": (
                                f"Hand ratio {hand_ratio:.4f} is below "
                                f"{args.min_hand_ratio:.4f}."
                            ),
                        }
                    )
                    continue
                pos, pres = preprocess_sequence(raw)
            except (OSError, ValueError, EOFError, IndexError) as exc:
                report_rows.append(
                    {
                        "split": split,
                        "video": r.video,
                        "label": r.gloss,
                        "status": "invalid_landmarks",
                        "frames": "",
                        "hand_ratio": "",
                        "error": str(exc),
                    }
                )
                continue
            P.append(pos)
            PR.append(pres)
            Y.append(g2i[r.gloss])
            PA.append(r.participant)
            VI.append(r.video)
            report_rows.append(
                {
                    "split": split,
                    "video": r.video,
                    "label": r.gloss,
                    "status": "processed",
                    "frames": len(raw),
                    "hand_ratio": hand_ratio,
                    "error": "",
                }
            )
        feature_path = C.FEATURES / f"{split}.npz"
        if feature_path.is_file():
            backup_path = feature_path.with_name(
                f"{feature_path.stem}."
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
                f".bak{feature_path.suffix}"
            )
            shutil.copy2(feature_path, backup_path)
            print(f"Existing {split} features backed up to {backup_path}")
        positions = (
            np.stack(P)
            if P
            else np.empty((0, C.SEQ_LEN, C.N_POINTS, 3), dtype=np.float32)
        )
        presence = (
            np.stack(PR)
            if PR
            else np.empty((0, C.SEQ_LEN, 2), dtype=np.float32)
        )
        with tempfile.NamedTemporaryFile(
            dir=feature_path.parent,
            prefix=f".{split}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
        try:
            np.savez_compressed(
                temporary_path,
                pos=positions,
                present=presence,
                y=np.asarray(Y, dtype=np.int64),
                participant=np.asarray(PA),
                video=np.asarray(VI),
                vocab=np.asarray(vocab),
                class_mapping_json=np.asarray(
                    json.dumps(build_class_mapping(vocab), ensure_ascii=False)
                ),
            )
            os.replace(temporary_path, feature_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        print(
            f"{split}: kept {len(Y)} of {len(df)} source clips; "
            f"{len(df) - len(Y)} excluded."
        )

    report_path = C.REPORTS / "preprocessing_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "split",
                "video",
                "label",
                "status",
                "frames",
                "hand_ratio",
                "error",
            ),
        )
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"Per-video preprocessing trace saved to {report_path}")


if __name__ == "__main__":
    main()
