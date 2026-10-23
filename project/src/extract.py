"""สกัด landmarks รายเฟรมด้วย MediaPipe Holistic Landmarker

เก็บไฟล์ .npy ต่อวิดีโอ รูปร่าง [T, N_POINTS, 3] (x, y, z แบบ normalized)
จุดที่ตรวจไม่เจอ = NaN (ห้ามใช้ 0 เพราะ 0 คือตำแหน่งจริงมุมภาพ)

ใช้งาน:
  python src/extract.py --workers 4                # ทำทั้ง subset_all.csv (ข้ามไฟล์ที่ทำแล้ว)
  python src/extract.py --limit 20                 # ทดลอง 20 วิดีโอก่อน
  python src/extract.py --probe path\to\video.mp4  # ดูโครงสร้างผลลัพธ์ของ MediaPipe
"""
import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C


def make_landmarker(model_path=C.MODEL_TASK):
    opts = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return mp.tasks.vision.HolisticLandmarker.create_from_options(opts)


def _lm(landmarks, idx_list):
    """แปลงรายการ landmark เป็น array [len(idx_list), 3]; ถ้าไม่เจอ = NaN"""
    out = np.full((len(idx_list), 3), np.nan, np.float32)
    if landmarks is None or len(landmarks) == 0:
        return out
    if isinstance(landmarks[0], (list, tuple)):  # เผื่อผลลัพธ์ซ้อนเป็นรายการของคน
        landmarks = landmarks[0]
    for k, i in enumerate(idx_list):
        if i < len(landmarks):
            p = landmarks[i]
            out[k] = (p.x, p.y, p.z)
    return out


def result_to_array(res):
    pose = _lm(getattr(res, "pose_landmarks", None), C.POSE_IDX)
    lh = _lm(getattr(res, "left_hand_landmarks", None), range(C.N_HAND))
    rh = _lm(getattr(res, "right_hand_landmarks", None), range(C.N_HAND))
    face = _lm(getattr(res, "face_landmarks", None), C.FACE_IDX)
    return np.concatenate([pose, lh, rh, face], axis=0)


def extract_video(video_path, out_path, max_side=640):
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": str(video_path), "frames": 0, "hand_ratio": 0.0, "status": "cannot_open"}
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0
    frames, last_ts, i = [], -1, 0
    with make_landmarker() as lmk:  # สร้างใหม่ทุกวิดีโอ เพื่อรีเซ็ตการติดตาม
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if max(h, w) > max_side:
                s = max_side / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            ts = max(last_ts + 1, int(i * 1000 / fps))
            last_ts = ts
            frames.append(result_to_array(lmk.detect_for_video(img, ts)))
            i += 1
    cap.release()
    if not frames:
        return {"video": str(video_path), "frames": 0, "hand_ratio": 0.0, "status": "no_frames"}
    arr = np.stack(frames).astype(np.float32)
    np.save(out_path, arr)
    hand = ~np.isnan(arr[:, C.LH, 0]).any(axis=1) | ~np.isnan(arr[:, C.RH, 0]).any(axis=1)
    return {"video": str(video_path), "frames": len(arr), "hand_ratio": float(hand.mean()), "status": "ok"}


def probe(video):
    """พิมพ์โครงสร้างผลลัพธ์ของเฟรมแรกๆ เพื่อตรวจว่า API ตรงกับที่โค้ดคาดไว้"""
    cap = cv2.VideoCapture(video)
    with make_landmarker() as lmk:
        for i in range(30):
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = lmk.detect_for_video(img, i * 33)
            if i in (0, 10, 29):
                print(f"frame {i}:")
                for name in ["pose_landmarks", "left_hand_landmarks", "right_hand_landmarks", "face_landmarks"]:
                    v = getattr(res, name, "MISSING ATTRIBUTE")
                    print(f"  {name}: {type(v).__name__}, len={len(v) if hasattr(v, '__len__') else '-'}")
                arr = result_to_array(res)
                print("  array", arr.shape, "NaN ratio", float(np.isnan(arr).mean()))


def _job(args):
    video, out = args
    try:
        return extract_video(video, out)
    except Exception as e:  # noqa: BLE001
        return {"video": str(video), "frames": 0, "hand_ratio": 0.0, "status": f"error: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--probe", type=str, default="")
    args = ap.parse_args()

    if not C.MODEL_TASK.exists():
        raise SystemExit("Model not found. Run: python src/download_model.py")
    if args.probe:
        probe(args.probe)
        return

    df = pd.read_csv(C.DATA / "subset_all.csv")
    videos_dir = C.find_videos_dir()
    C.LANDMARKS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for v in df.video:
        out = C.LANDMARKS / (v.rsplit(".", 1)[0] + ".npy")
        if not out.exists():
            jobs.append((videos_dir / v, out))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} videos to process ({len(df) - len(jobs)} already done or skipped)")

    qpath = C.LANDMARKS / "_quality.csv"
    new = not qpath.exists()
    t0 = time.time()
    with open(qpath, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["video", "frames", "hand_ratio", "status"])
        if new:
            wr.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_job, j) for j in jobs]
            for fut in tqdm(as_completed(futs), total=len(futs)):
                wr.writerow(fut.result())
                f.flush()
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":  # จำเป็นบน Windows (multiprocessing แบบ spawn)
    main()
