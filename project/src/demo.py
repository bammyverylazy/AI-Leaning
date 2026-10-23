"""เดโมเว็บแคม: กด SPACE เริ่ม/หยุดบันทึก 1 คำ แล้วโมเดลทำนาย

ปุ่ม: SPACE = เริ่ม/หยุดบันทึก   Q = ออก
ใช้งาน:
  python src/demo.py --ckpt models/gru_seed0.pt --thr 0.5 --camera 0
"""
import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

import config as C
from dataset import build_features, preprocess_sequence
from extract import make_landmarker, result_to_array
from model import SeqClassifier


def draw_text(img, text, org, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(C.MODELS / "gru_seed0.pt"))
    ap.add_argument("--thr", type=float, default=0.5, help="ต่ำกว่านี้ = ไม่แน่ใจ")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    a = ck["args"]
    model = SeqClassifier(ck["feature_dim"], len(ck["vocab"]), a["hidden"], 2, a["dropout"], a["kind"])
    model.load_state_dict(ck["model"])
    model.eval()
    vocab = ck["vocab"]

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # CAP_DSHOW เร็วกว่าบน Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("เปิดกล้องไม่ได้ ลอง --camera 1")

    recording, buf, t_start = False, [], 0.0
    status = "SPACE = start/stop recording"
    top = []
    lat_lm, lat_cls = [], []
    t_prev, fps = time.time(), 0.0
    t_begin, last_ts = time.time(), -1

    with make_landmarker() as lmk:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # กระจกเงา: ใช้แสดงผลเท่านั้น ไม่ได้ส่งเข้าโมเดล
            # สำคัญ: ส่งภาพที่ไม่กลับด้านเข้าโมเดล เพราะข้อมูลฝึกไม่ได้กลับด้าน
            raw_frame = cv2.flip(frame, 1)
            t0 = time.time()
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            ts = max(last_ts + 1, int((time.time() - t_begin) * 1000))  # ต้องเพิ่มขึ้นเสมอ
            last_ts = ts
            res = lmk.detect_for_video(img, ts)
            lat_lm.append((time.time() - t0) * 1000)
            arr = result_to_array(res)
            if recording:
                buf.append(arr)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            hands = (~np.isnan(arr[C.LH, 0]).any()) or (~np.isnan(arr[C.RH, 0]).any())
            draw_text(frame, f"FPS {fps:.0f}  landmarks {np.mean(lat_lm[-30:]):.0f} ms", (10, 25), 0.6)
            draw_text(frame, "REC" if recording else status, (10, 55), 0.7, (0, 0, 255) if recording else (255, 255, 255))
            if not recording and not hands:
                draw_text(frame, "no hands detected", (10, 85), 0.6, (0, 165, 255))
            for k, (w, p) in enumerate(top):
                col = (0, 255, 0) if k == 0 and p >= args.thr else (200, 200, 200)
                draw_text(frame, f"{k + 1}. {w}  {p * 100:.0f}%", (10, 120 + 30 * k), 0.7, col)
            cv2.imshow("ASL demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if not recording:
                    recording, buf, t_start = True, [], time.time()
                    top = []
                else:
                    recording = False
                    if len(buf) < 8:
                        status, top = "too short, try again", []
                        continue
                    t1 = time.time()
                    pos, pres = preprocess_sequence(np.stack(buf))
                    x = torch.from_numpy(build_features(pos, pres))[None]
                    with torch.no_grad():
                        prob = torch.softmax(model(x), 1)[0]
                    lat_cls.append((time.time() - t1) * 1000)
                    p, idx = prob.topk(3)
                    top = [(vocab[i], float(v)) for v, i in zip(p, idx)]
                    if top[0][1] < args.thr:
                        status = f"NOT SURE (best guess below {args.thr:.0%}) | classify {lat_cls[-1]:.0f} ms"
                    else:
                        status = f"classify {lat_cls[-1]:.0f} ms | {len(buf)} frames"

    cap.release()
    cv2.destroyAllWindows()
    if lat_lm:
        print(f"landmark latency: mean {np.mean(lat_lm):.1f} ms, p95 {np.percentile(lat_lm, 95):.1f} ms")
    if lat_cls:
        print(f"classifier latency: mean {np.mean(lat_cls):.1f} ms")


if __name__ == "__main__":
    main()
