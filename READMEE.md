# Sign Language Recognition — เอกสารเรียนรู้ช่วงที่ 1-4

โปรเจกต์: ระบบรู้จำคำภาษามือแบบแยกคำ ด้วย landmarks (MediaPipe) + โมเดลลำดับ (LSTM/GRU)

## ลำดับการเรียน

| ไฟล์ | หัวข้อ | เวลา |
|---|---|---|
| `notebooks/01_overview_pipeline.ipynb` | pipeline ทั้งระบบ, landmarks, normalize, ทำไมต้อง sequence model, data leakage | ~40 นาที |
| `notebooks/02_ml_basics.ipynb` | supervised/unsupervised/RL, train/val/test, linear regression + MSE + gradient descent, overfit, tree/forest/boosting, outlier, one-hot | ~1.5 ชม. |
| `notebooks/03_metrics_sklearn.ipynb` | confusion matrix, precision/recall/F1, macro-F1, threshold, confidence + "ไม่แน่ใจ", sklearn workflow | ~1 ชม. |
| `notebooks/04_deep_learning_basics.ipynb` | neuron → network, loss, backprop (เขียนเอง), Adam, lr/batch/epoch, อ่าน loss curve, overfit + วิธีแก้, PyTorch เทียบกัน | ~2 ชม. |

## วิธีรัน
```
pip install -r requirements.txt
jupyter notebook notebooks/
```
แต่ละ notebook มี **Mini challenge** ให้เปลี่ยนพารามิเตอร์แล้วสังเกตผล ทำแล้วจดลง experiment log

## สถานะการทดสอบ
- notebook 01-04: โค้ดทุกเซลล์ (ยกเว้น 2 เซลล์ PyTorch ในไฟล์ 04) **รันผ่านจริงแล้ว** ด้วย numpy 2.5 / pandas 3.0 / matplotlib 3.11 / scikit-learn 1.9
- เซลล์ PyTorch (04, ส่วนที่ 9) และ LightGBM/CatBoost (02, ในรูปโค้ดตัวอย่าง) **ยังไม่ได้รัน** เพราะติดตั้งไม่ได้ในสภาพแวดล้อมที่สร้าง — ตรวจแค่ไวยากรณ์
- ตัวเลขในข้อความอธิบายมาจากการรันครั้งที่สร้างไฟล์ ค่าของเธออาจต่างเล็กน้อยตามเวอร์ชันไลบรารี

## Resources ที่ค้นยืนยันแล้วว่ามีอยู่
**วิดีโอ / คอร์ส**
- 3Blue1Brown — Neural Networks (4 ตอนแรกครอบคลุม notebook 04 ส่วน 1-4): https://3blue1brown.com/neural-networks
- StatQuest (Josh Starmer) playlist "Machine Learning" — Bias and Variance, Cross Validation, Confusion Matrix, Sensitivity and Specificity, Decision Trees, Random Forests: https://www.classcentral.com/course/youtube-machine-learning-90293
- Andrej Karpathy — Neural Networks: Zero to Hero: https://github.com/karpathy/nn-zero-to-hero
- Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course

**อ่านเพิ่ม**
- Overfitting / loss curve (Google MLCC): https://developers.google.com/machine-learning/crash-course/overfitting/overfitting
- Adam + optimizer (D2L): https://classic.d2l.ai/chapter_optimization/adam.html
- sklearn `classification_report`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- Chris Olah's blog: http://colah.github.io/
- PyTorch tutorials: https://pytorch.org/tutorials

**GitHub repos**
- https://github.com/karpathy/nn-zero-to-hero
- https://github.com/jakevdp/PythonDataScienceHandbook (อ่านออนไลน์: https://jakevdp.github.io/PythonDataScienceHandbook/)
- https://github.com/mnielsen/neural-networks-and-deep-learning
- https://github.com/pytorch/tutorials

**จากเอกสารโจทย์ (ยังไม่ได้ใส่ลิงก์ เพราะไม่ได้ค้นยืนยัน — ค้นด้วยชื่อ):** MediaPipe Holistic Landmarker API, ASL Citizen dataset, WLASL paper
**ข้อควรจำ:** ASL Citizen / WLASL เป็นข้อมูล ASL ถ้าเลือกภาษามือไทยต้องหาชุดข้อมูลภาษามือไทยแยกต่างหาก

## ยังไม่ได้ทำ (ช่วงถัดไปตามแผน)
ช่วง 5-7 (PyTorch ให้คล่อง, RNN→LSTM/GRU, MediaPipe vs YOLO-pose, CNN/backbone/ViT แค่คอนเซ็ปต์) และช่วง 8 hands-on
