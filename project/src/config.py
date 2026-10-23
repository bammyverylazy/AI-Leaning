"""ค่าคงที่และ path ทั้งหมดของโปรเจกต์ (แก้ที่นี่ที่เดียว)"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
LANDMARKS = DATA / "landmarks"
FEATURES = DATA / "features"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

MODEL_TASK = MODELS / "holistic_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)

# ---- จุดที่เลือกเก็บต่อเฟรม ----
# Pose (33 จุดของ MediaPipe) เลือกเฉพาะท่อนบน: จมูก ตา หู ปาก ไหล่ ศอก ข้อมือ
POSE_IDX = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# Face (478 จุด) เลือกชุดย่อย: ปาก คิ้ว ตา จมูก โครงหน้า
FACE_IDX = (
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]  # ปาก
    + [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]  # คิ้ว
    + [33, 133, 159, 145, 362, 263, 386, 374]  # ตา
    + [1]  # ปลายจมูก
    + [10, 152, 234, 454]  # โครงหน้า
)

N_POSE = len(POSE_IDX)
N_HAND = 21
N_FACE = len(FACE_IDX)
N_POINTS = N_POSE + 2 * N_HAND + N_FACE  # ลำดับ: pose, มือซ้าย, มือขวา, face

POSE_SL = slice(0, N_POSE)
LH = slice(N_POSE, N_POSE + N_HAND)
RH = slice(N_POSE + N_HAND, N_POSE + 2 * N_HAND)
FACE_SL = slice(N_POSE + 2 * N_HAND, N_POINTS)

L_SH = POSE_IDX.index(11)  # ไหล่ซ้าย
R_SH = POSE_IDX.index(12)  # ไหล่ขวา

SEQ_LEN = 64  # จำนวนเฟรมหลัง resample


def find_split_csv(name: str) -> Path:
    """หาไฟล์ train.csv / val.csv / test.csv ใต้ data/raw"""
    hits = sorted(RAW.rglob(f"{name}.csv"), key=lambda p: ("split" not in str(p).lower(), len(str(p))))
    if not hits:
        raise FileNotFoundError(
            f"ไม่พบ {name}.csv ใต้ {RAW} - ตรวจว่าแตก ASL_Citizen.zip ไว้ใน data/raw แล้ว"
        )
    return hits[0]


def find_videos_dir() -> Path:
    """หาโฟลเดอร์ videos ของ ASL Citizen"""
    for p in RAW.rglob("videos"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"ไม่พบโฟลเดอร์ videos ใต้ {RAW}")
