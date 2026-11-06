"""realtime.py — Real-time ASL word-by-word subtitle (ไม่ต้องกด SPACE)

วิธีใช้:
  python src/realtime.py --ckpt models/gru_seed3.pt
  python src/realtime.py --ckpt models/gru_seed3.pt --thr 0.45 --camera 0 --no-landmarks

ปุ่ม:
  C = clear subtitle
  L = toggle landmark overlay
  Q / ESC = ออก

Pipeline:
  Webcam → MediaPipe Holistic (ทุกเฟรม)
           → sliding window [SEQ_LEN เฟรม]
           → preprocess (เหมือนตอน train)
           → BiGRU classifier
           → SubtitleManager (state machine)
           → แสดงผลบน OpenCV window
"""
from __future__ import annotations

import argparse
import collections
import time
from typing import Deque, List

import cv2
import mediapipe as mp
import numpy as np
import torch

import config as C
from dataset import build_features, preprocess_sequence
from extract import make_landmarker, result_to_array
from label_display import aggregate_predictions
from model import SeqClassifier
from model_vocabulary import checkpoint_vocabulary
from subtitle import State, SubtitleManager

# ---- ตัวเลขสีสำหรับ OpenCV (BGR) ----
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GREEN  = (50, 220, 50)
ORANGE = (20, 165, 255)
RED    = (50, 50, 220)
GRAY   = (180, 180, 180)
YELLOW = (0, 220, 220)
CYAN   = (220, 200, 0)
BLUE   = (220, 100, 20)

STATE_COLOR = {
    State.IDLE:      GRAY,
    State.CANDIDATE: ORANGE,
    State.CONFIRMED: GREEN,
    State.COOLDOWN:  CYAN,
}
STATE_LABEL = {
    State.IDLE:      "Waiting...",
    State.CANDIDATE: "Detecting",
    State.CONFIRMED: "Confirmed!",
    State.COOLDOWN:  "Pause",
}

INFER_EVERY = 3   # run classifier ทุก N เฟรม (ลด CPU)


def draw_text(img, text: str, org, scale=0.65, color=WHITE, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_filled_rect(img, x1, y1, x2, y2, color, alpha=0.55):
    sub = img[y1:y2, x1:x2]
    rect = np.full_like(sub, color)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub


def draw_progress_bar(img, x, y, w, h, progress: float, color=GREEN):
    cv2.rectangle(img, (x, y), (x + w, y + h), GRAY, 1)
    filled = int(w * min(progress, 1.0))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + h), color, -1)


def draw_landmark_overlay(frame, arr: np.ndarray):
    """วาด hand + pose landmarks บน frame (ขนาด H×W)"""
    H, W = frame.shape[:2]

    def dot(pts_slice, color, r=3):
        pts = arr[pts_slice]  # [P, 3]
        for p in pts:
            if not np.isnan(p[0]):
                cx, cy = int(p[0] * W), int(p[1] * H)
                cv2.circle(frame, (cx, cy), r, color, -1)

    dot(C.LH,      (0, 220, 255), 3)   # มือซ้าย → เหลือง
    dot(C.RH,      (0, 140, 255), 3)   # มือขวา → ส้ม
    dot(C.POSE_SL, (200, 200, 50), 2)  # pose → ฟ้าอ่อน
    # เชื่อมข้อต่อมือ (21 จุดตาม MediaPipe hand)
    _HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17),
    ]
    for hand_sl, col in [(C.LH, (0,220,255)), (C.RH, (0,140,255))]:
        pts = arr[hand_sl]
        for a, b in _HAND_CONNECTIONS:
            pa, pb = pts[a], pts[b]
            if not (np.isnan(pa[0]) or np.isnan(pb[0])):
                ax, ay = int(pa[0]*W), int(pa[1]*H)
                bx, by = int(pb[0]*W), int(pb[1]*H)
                cv2.line(frame, (ax,ay), (bx,by), col, 1, cv2.LINE_AA)


def wrap_text(text: str, font_scale: float, thickness: int, max_w: int) -> List[str]:
    """ตัดบรรทัด subtitle ให้พอดีกับความกว้าง"""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        (tw, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if tw <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def run(args):
    # ---- โหลดโมเดล ----
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    vocab = checkpoint_vocabulary(ck)
    model = SeqClassifier(ck["feature_dim"], len(vocab), a["hidden"], 2, a["dropout"], a["kind"])
    model.load_state_dict(ck["model"])
    model.eval()
    n_classes = len(vocab)
    print(f"Model loaded: {args.ckpt}  |  {n_classes} classes  |  feature_dim {ck['feature_dim']}")

    # ---- เปิด webcam ----
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"เปิดกล้อง {args.camera} ไม่ได้ ลอง --camera 1")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ---- state ----
    sub_mgr = SubtitleManager(
        min_votes=args.min_votes,
        conf_threshold=args.thr,
        cooldown_sec=args.cooldown,
    )
    frame_buf: Deque[np.ndarray] = collections.deque(maxlen=C.SEQ_LEN)
    show_landmarks = not args.no_landmarks

    # stats
    lat_lm: Deque[float] = collections.deque(maxlen=60)
    lat_cls: Deque[float] = collections.deque(maxlen=60)
    fps_meter: Deque[float] = collections.deque(maxlen=30)
    t_prev = time.time()

    # current prediction display
    cur_word = ""
    cur_conf = 0.0
    cur_original = ""
    cur_class_index = -1
    cur_top3 = []
    frame_count = 0
    t_begin = time.time()
    last_ts = -1

    with make_landmarker() as lmk:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            # ---- FPS ----
            now = time.time()
            fps_meter.append(1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            # ---- MediaPipe ----
            display_frame = cv2.flip(frame, 1)     # กระจกสำหรับแสดงผล
            raw_frame = frame                      # ไม่กลับด้าน → ส่ง MediaPipe

            t0 = time.time()
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            ts = max(last_ts + 1, int((now - t_begin) * 1000))
            last_ts = ts
            res = lmk.detect_for_video(mp_img, ts)
            lat_lm.append((time.time() - t0) * 1000)
            arr = result_to_array(res)  # [N_POINTS, 3]

            # flip x สำหรับ landmark overlay บน display_frame (เพราะ display_frame กลับแล้ว)
            arr_display = arr.copy()
            arr_display[:, 0] = 1.0 - arr_display[:, 0]

            hands_present = (
                not np.isnan(arr[C.LH, 0]).any()
                or not np.isnan(arr[C.RH, 0]).any()
            )
            frame_buf.append(arr)

            # ---- Inference ----
            newly_confirmed = None
            if frame_count % INFER_EVERY == 0 and len(frame_buf) == C.SEQ_LEN:
                t1 = time.time()
                raw_seq = np.stack(list(frame_buf))  # [T, N_POINTS, 3]
                pos, pres = preprocess_sequence(raw_seq)
                x = torch.from_numpy(build_features(pos, pres))[None]
                with torch.no_grad():
                    prob = torch.softmax(model(x), 1)[0]
                lat_cls.append((time.time() - t1) * 1000)
                cur_top3 = aggregate_predictions(prob, vocab, top_k=3)
                best = cur_top3[0]
                cur_word = best.canonical_label
                cur_conf = best.confidence
                cur_original = best.original_model_label
                cur_class_index = best.class_index

                # ส่งเข้า state machine
                newly_confirmed, _ = sub_mgr.update(
                    cur_word,
                    cur_conf,
                    hands_present,
                    original_model_label=cur_original,
                    class_index=cur_class_index,
                )
            elif not hands_present:
                sub_mgr.update(None, 0.0, False)

            # ---- วาด UI ----
            H, W = display_frame.shape[:2]

            if show_landmarks:
                draw_landmark_overlay(display_frame, arr_display)

            # --- Top bar (FPS + latency) ---
            fps = float(np.mean(fps_meter)) if fps_meter else 0.0
            lm_ms = float(np.mean(lat_lm)) if lat_lm else 0.0
            cls_ms = float(np.mean(lat_cls)) if lat_cls else 0.0
            draw_filled_rect(display_frame, 0, 0, W, 36, BLACK, 0.6)
            draw_text(display_frame, f"FPS {fps:.0f}  |  LM {lm_ms:.0f}ms  CLS {cls_ms:.0f}ms",
                      (10, 24), 0.6, GRAY)

            # --- State badge ---
            state = sub_mgr.state
            s_col = STATE_COLOR[state]
            s_label = STATE_LABEL[state]
            draw_filled_rect(display_frame, W - 160, 0, W, 36, BLACK, 0.6)
            draw_text(display_frame, s_label, (W - 150, 24), 0.6, s_col)

            # --- Current prediction panel ---
            panel_y = 44
            draw_filled_rect(display_frame, 0, panel_y, W, panel_y + 80, BLACK, 0.5)

            if not hands_present:
                draw_text(display_frame, "No hands detected", (10, panel_y + 28), 0.7, ORANGE)
            elif cur_top3:
                best_prediction = cur_top3[0]
                col = GREEN if best_prediction.confidence >= args.thr else ORANGE
                draw_text(
                    display_frame,
                    f"{best_prediction.canonical_label}  {best_prediction.confidence*100:.0f}%",
                    (10, panel_y + 28),
                    0.9,
                    col,
                )
                # top 2–3
                for k, prediction in enumerate(cur_top3[1:], 1):
                    draw_text(display_frame, f"  {k+1}. {prediction.canonical_label} {prediction.confidence*100:.0f}%",
                              (10, panel_y + 28 + 26 * k), 0.55, GRAY)

            # --- Candidate progress bar ---
            if state == State.CANDIDATE:
                cand = sub_mgr.candidate_word
                prog = sub_mgr.candidate_progress
                draw_text(display_frame, f"[{cand}]", (10, panel_y + 68), 0.6, YELLOW)
                draw_progress_bar(display_frame, 120, panel_y + 55, 200, 12, prog, GREEN)

            # --- Subtitle area (bottom) ---
            sub_text = sub_mgr.subtitle or "(subtitle will appear here)"
            sub_h = 90
            draw_filled_rect(display_frame, 0, H - sub_h, W, H, BLACK, 0.7)
            cv2.line(display_frame, (0, H - sub_h), (W, H - sub_h), (80, 80, 80), 1)
            draw_text(display_frame, "SUBTITLE:", (10, H - sub_h + 22), 0.5, GRAY, 1)
            lines = wrap_text(sub_text, 0.75, 2, W - 20)
            for li, line in enumerate(lines[-2:]):  # แสดงสูงสุด 2 บรรทัด
                draw_text(display_frame, line, (10, H - sub_h + 48 + li * 30), 0.75, WHITE)

            # --- Controls hint ---
            draw_text(display_frame, "C=clear  L=landmarks  Q=quit",
                      (W - 270, H - 8), 0.45, GRAY, 1)

            # --- Flash on new word ---
            if newly_confirmed:
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (W, H), GREEN, -1)
                cv2.addWeighted(overlay, 0.15, display_frame, 0.85, 0, display_frame)
                draw_text(display_frame, f"✓ {newly_confirmed}", (W//2 - 80, H//2), 1.2, GREEN)

            cv2.imshow("ASL Real-time", display_frame)

            # ---- Key handling ----
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                sub_mgr.clear()
            elif key == ord("l"):
                show_landmarks = not show_landmarks

    cap.release()
    cv2.destroyAllWindows()

    # ---- Summary ----
    print("\n--- Session Summary ---")
    print("Subtitle:", sub_mgr.subtitle)
    print("Words:", sub_mgr.word_list)
    print(
        "Original model labels:",
        [(entry.class_index, entry.original_model_label) for entry in sub_mgr.entries],
    )
    if lat_lm:
        print(f"Landmark latency: mean {np.mean(lat_lm):.1f} ms, p95 {np.percentile(list(lat_lm), 95):.1f} ms")
    if lat_cls:
        print(f"Classifier latency: mean {np.mean(lat_cls):.1f} ms")


def main():
    from application import main as launch_application

    launch_application()


if __name__ == "__main__":
    main()
