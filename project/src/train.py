"""ฝึก baseline GRU/LSTM แล้วประเมินบน test (ผู้แสดงที่ไม่เคยเห็น)

ใช้งาน:
  python src/train.py --kind gru --epochs 60 --seed 0
  python src/train.py --kind lstm --epochs 60 --seed 0
ผลลัพธ์: models/<kind>_seed<N>.pt, reports/<kind>_seed<N>_*.{json,png,csv}
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

import config as C
from dataset import FEATURE_DIM, SignDataset
from model import SeqClassifier


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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = [l.strip() for l in (C.DATA / "vocab.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    n_classes = len(vocab)

    train_ds = SignDataset(C.FEATURES / "train.npz", augment_data=True)
    val_ds = SignDataset(C.FEATURES / "val.npz")
    test_ds = SignDataset(C.FEATURES / "test.npz")
    # num_workers=0 ปลอดภัยบน Windows (ไม่ต้องกังวลเรื่อง spawn)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=len(train_ds) > args.batch)
    val_dl = DataLoader(val_ds, batch_size=128)
    test_dl = DataLoader(test_ds, batch_size=128)
    print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}  classes {n_classes}  device {device}")

    model = SeqClassifier(FEATURE_DIM, n_classes, args.hidden, 2, args.dropout, args.kind).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    C.MODELS.mkdir(exist_ok=True)
    C.REPORTS.mkdir(exist_ok=True)
    tag = f"{args.kind}_seed{args.seed}"
    ckpt = C.MODELS / f"{tag}.pt"
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
            torch.save({"model": model.state_dict(), "args": vars(args), "vocab": vocab, "feature_dim": FEATURE_DIM}, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    # ---- ประเมินด้วย checkpoint ที่ดีที่สุดบน validation ----
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    tl, ty = evaluate(model, test_dl, device)
    vl, vy = evaluate(model, val_dl, device)
    pred = tl.argmax(1).numpy()
    y_np = ty.numpy()
    metrics = {
        "top1": float(accuracy_score(y_np, pred)),
        "top5": topk_acc(tl, ty, 5),
        "macro_f1": float(f1_score(y_np, pred, average="macro")),
        "best_val_acc": best,
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
    (C.REPORTS / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

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
