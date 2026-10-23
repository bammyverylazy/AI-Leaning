# ASL Isolated Sign Recognition
### MediaPipe Holistic landmarks + BiGRU / BiLSTM classifier (webcam prototype)

An educational prototype that watches a short video (or webcam clip) of one American Sign Language (ASL) sign and predicts **which of 50 words** it is, with a **confidence score**, and answers **"not sure"** when confidence is low.

> This is a study prototype. It is **not** a communication tool for Deaf or hard-of-hearing people.

---

## Table of contents
1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [Scope](#2-scope)
3. [End-to-end pipeline (flowchart)](#3-end-to-end-pipeline-flowchart)
4. [Models used](#4-models-used)
5. [Data](#5-data)
6. [Step-by-step: what each script does](#6-step-by-step-what-each-script-does)
7. [The classifier in detail](#7-the-classifier-in-detail)
8. [Training and evaluation](#8-training-and-evaluation)
9. [The "not sure" mechanism](#9-the-not-sure-mechanism)
10. [Webcam demo](#10-webcam-demo)
11. [Glossary of technical terms](#11-glossary-of-technical-terms)
12. [How to run everything (Windows)](#12-how-to-run-everything-windows)
13. [Project structure](#13-project-structure)
14. [Results template](#14-results-template)
15. [Limitations](#15-limitations)
16. [Future work](#16-future-work)
17. [Troubleshooting](#17-troubleshooting)
18. [References and links](#18-references-and-links)

---

## 1. The idea in one minute

A computer does not "see" a hand like a human. A video is millions of colored pixels. Learning directly from pixels needs a lot of data and is easily confused by lighting, background and clothing.

So we split the problem in two:

```
 VIDEO ──► [Stage 1: find the body]  ──► sequence of coordinates ──► [Stage 2: understand the motion] ──► WORD + confidence
            MediaPipe (pre-trained)        (numbers only)             BiGRU / BiLSTM (we train this)
```

- **Stage 1 (pose detection)** turns every frame into a list of points: knuckles, fingertips, wrists, shoulders, mouth corners, eyebrows. Background, skin tone and clothes disappear. Only geometry remains.
- **Stage 2 (sign interpretation)** reads the sequence of those points over time and decides which sign it is.

The two stages are kept clearly separate: you can change or inspect each one without touching the other.

---

## 2. Scope

| Decision | Choice | Why |
|---|---|---|
| Sign language | ASL | Chosen target; large public dataset exists |
| Task | **Isolated** sign recognition (one word per clip) | Continuous sentence translation is a much harder, separate problem |
| Vocabulary | 50 words | Enough examples per word; expandable to 100 |
| Inputs | Hands + upper body + face subset | Signs use handshape, location, movement **and** facial expression |
| Output | Top-3 words, confidence, "not sure" option | Wrong-but-confident answers are worse than admitting uncertainty |
| Device | Windows laptop + webcam | Target for the demo and latency tests |

---

## 3. End-to-end pipeline (flowchart)

### 3.1 Whole project

```
┌────────────────────┐
│  ASL Citizen zip   │  83,399 videos, 2,731 signs, 52 signers (42.8 GB)
└─────────┬──────────┘
          │ fetch_subset.py  (downloads only the files we need)
          ▼
┌────────────────────┐
│  CSVs + videos     │  train.csv / val.csv / test.csv  +  ASL_Citizen/videos/*.mp4
└─────────┬──────────┘
          │ make_vocab.py  (pick 50 words, check signer overlap)
          ▼
┌────────────────────┐
│ vocab.txt          │  50 glosses; line number = class index
│ subset_*.csv       │  which videos belong to which split
└─────────┬──────────┘
          │ extract.py  (MediaPipe Holistic, frame by frame)
          ▼
┌────────────────────┐
│ data/landmarks/*.npy│  per video: [frames, 98 points, 3]  (NaN = not detected)
│ _quality.csv       │  % of frames with a detected hand
└─────────┬──────────┘
          │ dataset.py  (clean → normalize → resample → features)
          ▼
┌────────────────────┐
│ features/*.npz     │  train / val / test arrays
└─────────┬──────────┘
          │ train.py  (BiGRU or BiLSTM)
          ▼
┌────────────────────┐
│ models/*.pt        │  best checkpoint (by validation accuracy)
│ reports/*          │  metrics, curves, confusion matrix, per-signer accuracy
└─────────┬──────────┘
          │ demo.py
          ▼
     Webcam demo: word + confidence + "not sure"
```

### 3.2 Preprocessing of one video (`dataset.py`)

```
raw landmarks [T, 98, 3] with NaN
   │
   ├─► note which frames have left/right hand  ──────────────► "present" flags [T, 2]
   │
   ├─► trim leading/trailing frames with no hands (arms resting)
   ├─► interpolate NaN gaps in time
   ├─► normalize: subtract shoulder midpoint, divide by shoulder width
   ├─► resample to exactly 64 frames
   │
   ▼
positions [64, 98, 3]
   │
   ├─► velocity = difference between consecutive frames
   ▼
features [64, 590] = positions(294) + velocity(294) + present flags(2)
```

### 3.3 Inside the classifier

```
 features [B, 64, 590]                      B = batch size
        │
        ▼
 Linear(590→192) → LayerNorm → ReLU → Dropout
        │
        ▼
 Bidirectional GRU/LSTM, 2 layers, hidden 192      reads frames in time order
        │   output: [B, 64, 384]                    (384 = 2 directions × 192)
        ▼
 mean over time ⊕ max over time  → [B, 768]         summarize the whole clip
        │
        ▼
 Dropout → Linear(768→50) → logits
        │
        ▼
 softmax → probabilities (sum to 100%)  → top-3 + confidence
```

### 3.4 Live demo

```
 Webcam frame ─► MediaPipe Holistic ─► 98 points ─► (buffer while recording)
                                                         │ press SPACE to stop
                                                         ▼
                                  same preprocessing as training (preprocess_sequence)
                                                         ▼
                                       classifier ─► softmax ─► top-3
                                                         ▼
                                  confidence ≥ threshold ? show word : "NOT SURE"
```

---

## 4. Models used

There are exactly **two** model stages.

### 4.1 MediaPipe Holistic Landmarker (pre-trained, by Google, not trained by us)
- Input: one image frame. Output: landmark coordinates.
- Internally a bundle of smaller models: pose (BlazePose), face mesh, palm detection + hand landmarks.
- Full output is 553 landmarks (33 pose + 478 face + 21 per hand). **We keep 98.**
- Runs in `VIDEO` mode, which tracks across frames and needs strictly increasing timestamps.
- Model file: `holistic_landmarker.task` (float16).

### 4.2 BiGRU / BiLSTM sequence classifier (trained by us)
- Input: 64 frames × 590 features. Output: 50 class scores.
- We train **GRU** and **LSTM** versions and compare them (same architecture otherwise).

### What about Ultralytics / YOLO?
Suggested as an option. Standard YOLO-pose gives **17 body keypoints only: no fingers, no detailed face**, but ASL words are distinguished mainly by handshape. So it is not the main extractor. It could be an optional ablation. Note it is licensed AGPL-3.0, which matters if the project ever becomes commercial.

---

## 5. Data

**ASL Citizen** (Microsoft Research, Boston University, University of Washington)
- 83,399 videos, 2,731 signs, 52 signers (Deaf / hard-of-hearing), recorded on the contributors' own cameras with consent, which is close to real webcam conditions.
- Official splits are by **signer**.

**Our subset**
- 50 words chosen at random (fixed seed) from the words with enough examples in all splits (minimum 15 train, 3 val, 8 test).
- Videos kept: train 775, val 177, test 589 (1,541 total). About 15 training videos per word.
- Test set contains 11 signers the model never saw during training.
- Signer overlap between train/val/test is checked automatically and must be 0.

**Why split by signer (data leakage)**
If the same person appears in training and test, the model can memorize *that person's* style and look accurate while failing on anyone new. Splitting by signer measures what we actually care about: *does it work for a new person?*

---

## 6. Step-by-step: what each script does

| Script | What it does | Key idea |
|---|---|---|
| `config.py` | Paths, which landmark indices to keep, sequence length (64) | One place to change settings |
| `download_model.py` | Downloads `holistic_landmarker.task` | Model file for Stage 1 |
| `fetch_subset.py` | Reads the zip's file list remotely and extracts only needed files (HTTP range requests) | Avoids a 42.8 GB download |
| `make_vocab.py` | Reads the split CSVs, picks 50 words, writes `vocab.txt` + `subset_*.csv`, checks signer overlap | Reproducible vocabulary |
| `extract.py` | Runs MediaPipe on every frame, saves `[T, 98, 3]` per video, logs hand-detection ratio | Slowest step; parallel and resumable |
| `dataset.py` | Cleans, normalizes, resamples, builds features, defines augmentation and the PyTorch `Dataset` | Turns raw points into learnable input |
| `model.py` | Defines the BiGRU/BiLSTM classifier | Stage 2 |
| `train.py` | Trains, picks best epoch on validation, evaluates on test, writes reports | Honest evaluation |
| `demo.py` | Webcam recording → prediction → display, with latency and FPS | Live prototype |

### The 98 selected points (`config.py`)
- **Pose: 13** (nose, eyes, ears, mouth corners, shoulders, elbows, wrists)
- **Left hand: 21**, **right hand: 21** (wrist, thumb, 4 fingers × 4 joints)
- **Face: 43** (mouth 20, eyebrows 10, eyes 8, nose tip 1, face outline 4)

Coordinates are `(x, y, z)`. `x, y` are normalized to the image (0–1); `z` is relative depth.

### Why `NaN` and not `0` for missing points
If a hand is hidden, MediaPipe returns nothing. Writing `0` would claim "the hand is at the image corner", which is a real position and would teach the model a lie. `NaN` means "unknown" and is handled explicitly (interpolation, plus a "hand present" flag as a feature).

---

## 7. The classifier in detail

### Why a sequence model?
A sign is **motion over time**. A single frame rarely tells you the word; the trajectory does. So we need a model that reads frames **in order** and remembers the past.

### RNN, LSTM, GRU
- **RNN (Recurrent Neural Network):** processes one time step at a time and passes a "hidden state" (its memory) to the next step. Plain RNNs forget long-range information (the *vanishing gradient* problem).
- **LSTM (Long Short-Term Memory):** adds **gates** that learn what to *forget*, what to *store*, and what to *output*, plus a separate long-term **cell state**. Better at keeping important information across many frames.
- **GRU (Gated Recurrent Unit):** a simpler cousin with fewer gates (update and reset) and no separate cell state. Fewer parameters, trains faster, often performs similarly.
- **Bidirectional (Bi):** two RNNs, one reading forward and one backward, and their outputs are joined, so every frame has context from both before and after.

Analogy: watching a clip while taking notes. LSTM/GRU decide which notes to keep, rewrite or throw away; at the end you decide the word from the notes.

### Our layers
1. `Linear(590→192) + LayerNorm + ReLU + Dropout(0.3)`: compress and rescale each frame's features.
2. `BiGRU/BiLSTM`, 2 layers, hidden size 192: read the sequence.
3. **Mean pooling + max pooling over time:** mean = overall motion; max = the strongest moments. Concatenated → 768 numbers.
4. `Dropout → Linear(768→50)`: one score (logit) per word.
5. `softmax`: turn scores into probabilities.

---

## 8. Training and evaluation

### Training setup
| Item | Value |
|---|---|
| Loss | Cross-entropy with **label smoothing 0.1** |
| Optimizer | **AdamW**, lr 1e-3, weight decay 1e-2 |
| LR schedule | Cosine annealing |
| Batch size / epochs | 64 / up to 60 |
| Regularization | Dropout 0.3, weight decay, gradient clipping (1.0), augmentation |
| Early stopping | Patience 12 on validation accuracy; keeps best checkpoint |
| Seeds | Set via `--seed`; repeat with several seeds and report mean ± std |

### Augmentation (training set only)
Time warping (speed changes), rotation ±10°, scale 0.9–1.1, small shifts, small Gaussian noise. These imitate different speeds, camera tilt and distance. Mirroring is **not** used (it would swap left/right hands).

### Metrics (computed on the unseen-signer test set)
- **Top-1 accuracy:** correct word is the first guess.
- **Top-5 accuracy:** correct word is in the five best guesses.
- **Macro-F1:** F1 per word, averaged equally, so easy words cannot hide hard ones.
- **Confusion matrix:** rows = true word, columns = predicted word. Off-diagonal cells show which signs get mixed up.
- **Per-signer accuracy:** shows whether some people are much harder than others.
- **Latency:** milliseconds for landmarks and for classification, measured in the demo.

### Train / validation / test roles
- **Train:** the model learns from it.
- **Validation:** used to choose the best epoch and the confidence threshold.
- **Test:** touched once at the end; the honest score on new people.

### Overfitting
When training loss keeps falling but validation loss/accuracy stops improving or worsens, the model is memorizing rather than generalizing. With only about 15 videos per word, this is a real risk. Countermeasures: augmentation, dropout, weight decay, early stopping, a small model.

---

## 9. The "not sure" mechanism

1. The softmax gives a probability for every word; the highest is the **confidence**.
2. On the **validation** set, `train.py` searches for the smallest threshold where the answers kept are at least 90% accurate while still answering at least 50% of clips.
3. If no threshold satisfies this, the threshold is 0 (no rejection). Check `reports/*_metrics.json`.
4. On test, we report **coverage** (fraction answered) and **accuracy on the answered part**.
5. In the demo, confidence below the threshold shows **NOT SURE**.

Softmax confidence is only a rough signal and can still be wrong on unusual inputs. It is not a guarantee.

---

## 10. Webcam demo

- Press **SPACE** to start recording a sign, **SPACE** again to stop and predict, **Q** to quit.
- Landmarks are extracted live, buffered, then processed with exactly the same `preprocess_sequence` and `build_features` as in training. Using identical preprocessing is essential.
- The screen shows FPS, landmark latency, top-3 predictions with percentages, and "no hands detected" warnings.
- The image sent to the model is **not mirrored** (only the display is), to match the training data orientation.
- On exit it prints mean and p95 landmark latency and mean classifier latency.

Things to test and record: bright vs dim light, plain vs busy background, camera height and distance, hands partly out of frame, several different people.

---

## 11. Glossary of technical terms

| Term | Meaning |
|---|---|
| **Computer vision** | Getting information from images or video |
| **Pixel** | One colored dot in an image; images are grids of numbers |
| **Frame** | One image in a video |
| **ISLR** | Isolated Sign Language Recognition: one sign per clip |
| **Gloss** | The written label for a sign (e.g. `HELLO`) |
| **Landmark / keypoint** | A named point on the body (knuckle, shoulder, mouth corner) |
| **Pose estimation** | Predicting landmark positions from an image |
| **Holistic** | Body, hands and face together |
| **Normalized coordinates** | Positions as fractions of image width/height (0–1) |
| **NaN** | "Not a number": marks a missing value |
| **Interpolation** | Filling gaps by estimating values from neighbors in time |
| **Normalization** | Rescaling data to a common reference (here: shoulder-centered, shoulder-width scale) |
| **Resampling** | Changing the number of frames (every clip becomes 64) |
| **Feature** | A number the model uses as input |
| **Velocity feature** | Frame-to-frame difference; encodes motion directly |
| **Sequence model** | A model that processes ordered data |
| **RNN / LSTM / GRU** | Recurrent networks with memory (see section 7) |
| **Hidden state** | The network's running memory vector |
| **Gate** | A learned 0–1 valve that controls what to keep or forget |
| **Bidirectional** | Reads the sequence forward and backward |
| **Pooling** | Summarizing a sequence into one vector (mean / max) |
| **Logit** | Raw score before softmax |
| **Softmax** | Converts scores to probabilities that sum to 1 |
| **Confidence** | The highest softmax probability |
| **Loss (cross-entropy)** | How wrong the predictions are; training minimizes it |
| **Label smoothing** | Softens the targets slightly to reduce overconfidence |
| **Optimizer (AdamW)** | The algorithm that updates weights using gradients |
| **Learning rate** | Step size of each update |
| **Weight decay** | Penalty on large weights; a regularizer |
| **Gradient clipping** | Caps gradient size for stable training |
| **Backpropagation** | Computing how each weight affects the loss |
| **Epoch** | One pass over the training set |
| **Batch** | A group of samples processed together |
| **Dropout** | Randomly switching off units during training to reduce memorization |
| **Overfitting** | Great on training data, poor on new data |
| **Regularization** | Anything that reduces overfitting |
| **Early stopping** | Stop when validation stops improving |
| **Augmentation** | Creating varied copies of training data (speed, rotation, scale, noise) |
| **Data leakage** | Test information sneaking into training, making scores look too good |
| **Top-k accuracy** | Correct answer is among the k best guesses |
| **Precision / Recall / F1** | Correctness of positive predictions / coverage of true positives / their balance |
| **Macro-F1** | F1 averaged equally across classes |
| **Confusion matrix** | Table of true vs predicted classes |
| **Coverage** | Fraction of inputs the system chooses to answer |
| **Latency** | Time from input to output (ms) |
| **FPS** | Frames per second |
| **Checkpoint** | Saved model weights |
| **Seed** | Fixed random start so results can be reproduced |
| **Ablation** | Changing one component to measure its effect |

---

## 12. How to run everything (Windows)

Run from the project folder in PowerShell, with the virtual environment active.

```powershell
# 0. setup (Python 3.11 or 3.12 recommended)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install remotezip

# 1. model file
python src/download_model.py

# 2. data (downloads only what we need)
python src/fetch_subset.py list      # check connection / see file names
python src/fetch_subset.py csv       # CSVs only
python src/make_vocab.py --n 50      # choose 50 words
python src/fetch_subset.py videos    # only those words' videos

# 3. landmarks
$v = (Get-ChildItem data\raw\ASL_Citizen\videos -Filter *.mp4 | Select-Object -First 1).FullName
python src/extract.py --probe $v     # sanity check of MediaPipe output
python src/extract.py --limit 20     # small trial, note the time
python src/extract.py --workers 4    # everything (resumable)

# 4. features and training
python src/dataset.py --min-hand-ratio 0.2
python src/train.py --kind gru  --seed 0
python src/train.py --kind lstm --seed 0

# 5. demo
python src/demo.py --ckpt models\gru_seed0.pt --thr 0.5
```

Download links:
- Holistic Landmarker model: https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
- ASL Citizen (full): https://www.microsoft.com/en-us/download/details.aspx?id=105253

---

## 13. Project structure

```
asl_project/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ config.py          paths, landmark indices, constants
│  ├─ download_model.py  get holistic_landmarker.task
│  ├─ fetch_subset.py    partial download from the remote zip
│  ├─ make_vocab.py      choose words, split CSVs, signer-overlap check
│  ├─ extract.py         MediaPipe → landmarks .npy per video
│  ├─ dataset.py         preprocessing, augmentation, Dataset
│  ├─ model.py           BiGRU / BiLSTM classifier
│  ├─ train.py           training + evaluation + reports
│  └─ demo.py            webcam demo
├─ data/
│  ├─ raw/               ASL Citizen CSVs + videos
│  ├─ landmarks/         per-video landmark arrays + _quality.csv
│  ├─ features/          train/val/test .npz
│  ├─ vocab.txt          the 50 words
│  └─ subset_*.csv       chosen videos per split
├─ models/               holistic_landmarker.task, trained *.pt
└─ reports/              metrics, curves, confusion matrices, per-signer accuracy
```

Files written by `train.py` in `reports/` (for a run named e.g. `gru_seed0`):
`*_metrics.json`, `*_curves.png`, `*_confusion.png`, `*_top_confusions.csv`, `*_per_signer.csv`.

---

## 14. Results template

Fill these in from your own runs (do not copy numbers from other papers).

| Model | Seeds | Top-1 | Top-5 | Macro-F1 | Threshold | Coverage | Acc. on answered |
|---|---|---|---|---|---|---|---|
| BiGRU | | | | | | | |
| BiLSTM | | | | | | | |

Latency on the laptop: landmarks ___ ms/frame, classifier ___ ms/clip, FPS ___.

Error analysis checklist: the most confused word pairs (`*_top_confusions.csv`), why (similar handshape? same location? hand occluded?), accuracy per signer, accuracy versus hand-detection ratio.

For context only: the ASL Citizen paper reports about 63% accuracy over **all 2,731 signs**. That is a different, harder task than 50 words, so the numbers are not directly comparable.

---

## 15. Limitations

- **Little data per word** (about 15 training videos), so results can vary a lot between runs.
- **Isolated signs only**; you press a key to mark start and end. No continuous translation.
- **Landmark errors** from poor light, motion blur, strong angles, hands covering the face or each other, or fast fingers.
- **Domain gap:** dataset recording conditions differ from your own webcam setup.
- **Signer differences:** speed, style, dominant hand, regional variation.
- **Sign language is more than hands:** meaning can depend on facial expression, body shifts and context; we use only a subset of face points.
- **Confidence is approximate.**
- **No evaluation with Deaf signers** beyond the dataset; the system has not been validated for real communication.
- **Dataset terms apply:** check the ASL Citizen license before any use beyond study (it is released for non-commercial research).

---

## 16. Future work

- More words and more data per word.
- Other architectures: 1D-CNN, small Transformer, graph networks on the skeleton.
- Compare other extractors (e.g. Ultralytics hand-keypoint models) as an ablation.
- Automatic start/end detection of a sign.
- Continuous sign recognition and then translation.
- Evaluate with Deaf signers and involve the community in design decisions.
- Export to ONNX / mobile for on-device use.

---

## 17. Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `fetch_subset.py : not recognized` | Run it with Python: `python src/fetch_subset.py list` |
| `pip install mediapipe` fails | Use Python 3.11 or 3.12 |
| Model file not found | Run `python src/download_model.py` or download the link and place it in `models\` |
| `MISSING ATTRIBUTE` in `--probe` | MediaPipe output names differ; send the printed output to adjust `result_to_array` |
| Extraction very slow or PC freezes | Lower `--workers` to 2 |
| `No such file ... landmarks` in `dataset.py` | Extraction not finished; rerun `extract.py` (skips finished files) |
| Very few samples kept | Lower `--min-hand-ratio` and check `data/landmarks/_quality.csv` |
| Camera does not open | Try `--camera 1`; close other apps using the camera |
| Poor webcam accuracy but good test accuracy | Domain gap: improve lighting, distance, plain background; sign at similar speed |
| Everything is "NOT SURE" | Threshold too high for your setup; try `--thr 0.3` and check hand detection |

---

## 18. References and links

- Desai et al., **ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition**, 2023. https://arxiv.org/abs/2304.05934
- ASL Citizen project page: https://www.microsoft.com/en-us/research/project/asl-citizen/
- MediaPipe Holistic Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker
- Li et al., **WLASL** (word-level ASL benchmark): used as a reference for benchmarking, not as training data here.
- Hochreiter & Schmidhuber, *Long Short-Term Memory*, 1997 (LSTM). Cho et al., 2014 (GRU).
