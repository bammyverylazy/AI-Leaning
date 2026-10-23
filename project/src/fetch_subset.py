import sys, pandas as pd
from remotezip import RemoteZip
import config as C

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
step = sys.argv[1] if len(sys.argv) > 1 else "csv"

with RemoteZip(URL) as z:
    names = z.namelist()
    if step == "list":                      # ดูโครงสร้างไฟล์ก่อน
        print(len(names)); print("\n".join(names[:15])); sys.exit()
    if step == "csv":                       # ขั้น 1: ดึงเฉพาะ CSV
        for n in names:
            if n.lower().endswith(".csv"):
                z.extract(n, C.RAW); print("got", n)
        sys.exit()
    # ขั้น 2: ดึงเฉพาะวิดีโอของคำที่เลือกไว้ใน subset_all.csv
    df = pd.read_csv(C.DATA / "subset_all.csv")
    wanted = set(df.video)
    todo = [n for n in names if n.rsplit("/", 1)[-1] in wanted]
    print(len(todo), "videos")
    for i, n in enumerate(todo, 1):
        if not (C.RAW / n).exists():
            z.extract(n, C.RAW)
        if i % 50 == 0: print(i, "/", len(todo))