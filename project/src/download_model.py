"""ดาวน์โหลดไฟล์โมเดล holistic_landmarker.task (ประมาณหลายสิบ MB)"""
import urllib.request

import config as C


def main():
    C.MODELS.mkdir(parents=True, exist_ok=True)
    if C.MODEL_TASK.exists() and C.MODEL_TASK.stat().st_size > 1_000_000:
        print("Already exists:", C.MODEL_TASK)
        return
    print("Downloading", C.MODEL_URL)
    urllib.request.urlretrieve(C.MODEL_URL, C.MODEL_TASK)
    print("Saved to", C.MODEL_TASK, f"({C.MODEL_TASK.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
