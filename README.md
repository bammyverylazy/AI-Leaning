# Sign Language Learning Project

Goal: webcam -> hand -> skeleton -> sign -> text -> speech. Learning first.

## Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate     Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```
Download `hand_landmarker.task` (MediaPipe Hand Landmarker, from the Google AI Edge docs) into `models/`.
Put a photo of your hand at `data/myhand.jpg`.

## Order
| Phase | Notebook | You learn |
|---|---|---|
| 0 | 01_python_cv_basics | pixels, arrays, OpenCV |
| 1 | 04_hand_detection | box, confidence, IoU |
| 1 | 05_hand_keypoints | skeleton, 21 numbers, normalization, live webcam |
| 2 | 03_dataset_exploration | record data, imbalance, leakage, splits |
| 2 | 06_baseline_classifier | KNN / LogReg, confusion matrix |

Folders: `data/` (signs/, processed/, splits.csv), `models/`, `experiments/`.
