"""เลือกชุดคำ (gloss) N คำจาก ASL Citizen และสร้างไฟล์ subset

ผลลัพธ์:
  data/vocab.txt            รายชื่อคำ (ลำดับบรรทัด = label index)
  data/subset_train.csv     คอลัมน์: participant, video, gloss
  data/subset_val.csv
  data/subset_test.csv
  data/subset_all.csv       รวมทุก split (มีคอลัมน์ split)

ตัวอย่าง:
  python src/make_vocab.py --n 50
"""
import argparse
import random

import pandas as pd

import config as C


def load_split(path):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        raise KeyError(f"ไม่พบคอลัมน์ {names} ใน {path}; คอลัมน์ที่มี: {list(df.columns)}")

    return pd.DataFrame(
        {
            "participant": df[pick("participant id", "participant", "signer", "user")].astype(str),
            "video": df[pick("video file", "video", "filename", "file")].astype(str),
            "gloss": df[pick("gloss", "sign", "label")].astype(str),
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="จำนวนคำ")
    ap.add_argument("--min-train", type=int, default=15)
    ap.add_argument("--min-val", type=int, default=3)
    ap.add_argument("--min-test", type=int, default=8)
    ap.add_argument("--pick", choices=["random", "top"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = {s: load_split(C.find_split_csv(s)) for s in ["train", "val", "test"]}
    for s, df in splits.items():
        print(f"{s}: {len(df)} videos, {df.participant.nunique()} signers, {df.gloss.nunique()} glosses")

    # ตรวจว่าผู้แสดงไม่ซ้ำข้าม split (กัน data leakage)
    sets = {s: set(df.participant) for s, df in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = sets[a] & sets[b]
        print(f"signer overlap {a}/{b}: {len(inter)}", "OK" if not inter else f"WARNING {sorted(inter)[:5]}")

    counts = pd.DataFrame({s: df.gloss.value_counts() for s, df in splits.items()}).fillna(0).astype(int)
    ok = counts[(counts.train >= args.min_train) & (counts.val >= args.min_val) & (counts.test >= args.min_test)]
    print(f"eligible glosses: {len(ok)}")
    if len(ok) < args.n:
        print(f"WARNING: only {len(ok)} eligible, using all. Lower the --min-* values for more.")
        chosen = sorted(ok.index)
    elif args.pick == "top":
        chosen = sorted(ok.sort_values("train", ascending=False).index[: args.n])
    else:
        chosen = sorted(random.Random(args.seed).sample(sorted(ok.index), args.n))

    C.DATA.mkdir(parents=True, exist_ok=True)
    (C.DATA / "vocab.txt").write_text("\n".join(chosen) + "\n", encoding="utf-8")
    frames = []
    for s, df in splits.items():
        sub = df[df.gloss.isin(chosen)].reset_index(drop=True)
        sub.to_csv(C.DATA / f"subset_{s}.csv", index=False)
        frames.append(sub.assign(split=s))
        print(f"subset_{s}: {len(sub)} videos, {sub.participant.nunique()} signers")
    pd.concat(frames).to_csv(C.DATA / "subset_all.csv", index=False)
    print("Saved vocab.txt and subset_*.csv in", C.DATA)


if __name__ == "__main__":
    main()
