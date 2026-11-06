# ASL Isolated Sign Recognition

An educational desktop prototype for recognizing one of 50 selected American
Sign Language (ASL) signs from a webcam. It combines MediaPipe Holistic
landmarks, a temporal GRU classifier, prediction stabilization, and a
word-level subtitle history.

> **Scope:** this is isolated sign recognition, not unrestricted continuous
> sign-language understanding or English translation. The offline test results
> reported here are from the held-out ASL Citizen subset; they are not webcam
> accuracy claims.

## Contents

1. [Project overview](#1-project-overview)
2. [Problem and motivation](#2-problem-and-motivation)
3. [System architecture](#3-system-architecture)
4. [Data and model pipeline](#4-data-and-model-pipeline)
5. [Landmarks and feature processing](#5-landmarks-and-feature-processing)
6. [Sequence model](#6-sequence-model)
7. [Real-time recognition](#7-real-time-recognition)
8. [Prediction stabilization](#8-prediction-stabilization)
9. [Subtitle generation](#9-subtitle-generation)
10. [Testing Mode](#10-testing-mode)
11. [Evaluation method and current results](#11-evaluation-method-and-current-results)
12. [Improvements in the current implementation](#12-improvements-in-the-current-implementation)
13. [Current project status](#13-current-project-status)
14. [Current data inventory](#14-current-data-inventory)
15. [How to run the project](#15-how-to-run-the-project)
16. [Presentation demo flow](#16-presentation-demo-flow)
17. [Presentation talking points](#17-presentation-talking-points)
18. [Limitations](#18-limitations)
19. [Future work](#19-future-work)

## 1. Project overview

The project recognizes a **single isolated sign at a time** from an ordered
sequence of webcam frames. MediaPipe converts each frame into body, hand, and
face landmarks. A trained sequence classifier maps a fixed-length feature
sequence to scores for 50 vocabulary classes. The desktop application displays
the current ranked predictions and commits sufficiently stable predictions to
a chronological word-level subtitle history.

The repository also includes:

- an offline ASL Citizen data and model-training pipeline;
- a guided Testing Mode that records prompted webcam attempts and supports
  separate correctness review;
- a Self-recording Studio that stores recordings for technical review,
  independent label verification, and controlled split assignment;
- model, dataset, and Testing Mode audit/report utilities.

**Presentation summary:** “This project is a webcam-based ASL isolated-sign
recognition prototype. It turns video into normalized landmark motion features,
classifies a 64-frame sequence, and uses a separate stabilization state machine
before adding a word to the subtitle history.”

### What the project does not claim

- It does not translate unrestricted, continuous signing into grammatical
  English sentences.
- It does not perform language understanding, grammar correction, or
  sentence-level translation.
- A top prediction or confidence score is not proof that the sign was performed
  correctly. Softmax scores are model scores, not a guarantee of calibrated
  probability.
- Results on ASL Citizen do not establish accuracy for arbitrary signers,
  cameras, lighting, or real-world conversations.

## 2. Problem and motivation

Signs are visual actions, not just hand poses. Hand shape, position relative to
the body, movement over time, and sometimes facial/body cues can distinguish
signs. A webcam system also has to cope with missed landmarks, framing and
lighting changes, different signing styles, and predictions that change from
frame to frame.

A classifier's latest label is not automatically a useful subtitle. A raw
prediction may be low-confidence or flicker between signs; repeatedly showing
it as confirmed text would be hard to read. The project therefore treats
recognition as a pipeline:

```text
Data and labels
      → landmark extraction
      → shared preprocessing and feature construction
      → sequence-model inference
      → prediction stabilization
      → subtitle history
      → controlled evaluation
```

Each stage addresses a different source of error. In particular, reliable
training data needs both technically usable sequences and labels that represent
the sign actually performed—not merely the prompt shown to the signer.

## 3. System architecture

```text
OFFLINE TRAINING
ASL Citizen metadata / selected videos
      → MediaPipe Holistic
      → per-video landmark arrays
      → normalize, interpolate, trim, resample
      → positions + velocities + hand-presence features
      → train GRU/LSTM candidate and evaluate on fixed splits

LIVE APPLICATION
Webcam frame
      → MediaPipe Holistic landmarks
      → 64-frame rolling buffer
      → same preprocessing and feature builder
      → GRU checkpoint
      → class probabilities and ranked labels
      → confidence / vote / cooldown state machine
      → chronological word-level subtitle history

SELF-RECORDED DATA
Webcam recording
      → technical validation and feature snapshot
      → pending review
      → accept or reject
      → independently verify performed sign
      → assign signer-disjoint split
      → export and explicitly opt in during candidate training
```

| Stage | Input | Work performed | Output and purpose |
|---|---|---|---|
| Landmark detection | Video frames | MediaPipe Holistic estimates selected body, hand, and face points. | Per-frame coordinates; reduces pixels to a structured representation. |
| Preprocessing | Landmark sequence, including missing points | Derives hand-presence flags, trims idle edges, interpolates missing coordinates, normalizes, and resamples. | Consistent 64-frame positions for clips of different lengths. |
| Feature construction | Normalized positions and hand flags | Flattens positions, adds frame-to-frame differences, and appends the two hand flags. | A 64 × 590 float feature sequence. |
| Classification | Feature sequence | The checkpoint returns class logits, converted to ranked probabilities. | Candidate signs and confidence scores. |
| Stabilization | Candidate label, score, and hand-presence signal | Requires repeated sufficiently confident predictions and an idle/cooldown period. | A committed word or no commitment. |
| Subtitle display | Committed words | Appends canonical labels in order and bounds the history length. | Readable word-level output, not translated sentences. |
| Evaluation | Known test label or independently reviewed ground truth | Compares predictions with the appropriate reference and reports metrics. | Evidence about a specified test set, not a universal accuracy claim. |

## 4. Data and model pipeline

### 4.1 Data sources

| Data source | Present use | Important distinction |
|---|---|---|
| ASL Citizen | The active 50-label subset supplies the existing train, validation, and test feature archives. | The splits are organized by signer. These are the data behind the existing offline model metrics. |
| Self-recorded Studio samples | Persisted separately under `project/data/self_recorded/` with an SQLite record and NPZ sequence artifacts. | Accepted recordings are not automatically training examples. Current samples are not eligible because none has verified ground truth or an assigned split. |
| Guided Testing Mode | Stores prompted webcam attempt records in `project/data/recognition_attempts.sqlite3`; stabilized attempts may also save the exact model-input sequence under `project/data/evaluation/sequences/`. | A target prompt is not verified truth. These attempts are not silently mixed into training. |
| Sign demonstration catalog | `project/data/sign_demos/manifest.csv` defines optional vocabulary-matched reference videos. | The current manifest marks all 50 entries as missing; there are currently no verified reference videos available to watch. These are reference demonstrations, not the offline test set. |
| Legacy recorder | `src/record_self.py` writes MP4 clips and a CSV manifest under the raw-data tree. | This older path does not create samples in the current reviewed SQLite store and its `--split` value does not satisfy the current training gate. |

Data provenance matters because a result is only interpretable when its source,
signer, session, label, preprocessing, and split are known. Training and
evaluation should also avoid putting clips from the same signer on both sides
of a split.

### 4.2 Self-recording and metadata

The current in-app Self-recording Studio flow is:

```text
Choose a vocabulary target and start a signer/session
      → record webcam frames
      → extract landmarks and build model features
      → inspect the preview and technical checks
      → save a pending sample, or discard the preview
      → review and accept/reject
      → independently verify the sign actually performed
      → assign a split and export only when eligible
```

The store records fields including target label, ground-truth label, signer and
session IDs, recording time, dominant hand, camera source/index, camera angle,
distance, lighting, signing speed, notes, frame/duration counts, landmark
detection/quality information, preprocessing version, model version, sequence
path, raw-video path when available, sequence hash, review status, reviewer,
and dataset split. Fields are subject to validation; this is not a claim that
every field describes a scientifically controlled collection.

- A **signer ID** groups one person's signing style so that the same signer can
  be kept out of a held-out split.
- A **session ID** groups recordings from one collection session and supports
  traceability; it does not by itself prevent leakage.
- IDs should be anonymous/pseudonymous, not real names.

### 4.3 Review and ground truth

The target prompt, model prediction, and verified ground truth are separate
values:

| Value | Meaning |
|---|---|
| `target_label` | The sign the recording workflow asked the signer to perform. |
| Model prediction | The class selected by the checkpoint; this may be wrong. |
| `ground_truth_label` | The sign independently observed in the recording and entered by a reviewer. |

The collection store uses a database `status` of `pending_review`, `accepted`,
or `rejected`, alongside explicit recording, verification, and quality fields:

| State / field | Meaning |
|---|---|
| `pending_review` / `PENDING_REVIEW` | Saved and awaiting a human review decision. |
| `accepted` / `ACCEPTED_UNVERIFIED` | Accepted after review, but not yet assigned verified truth. |
| `rejected` / `REJECTED` | Rejected, with a reason required by the review flow. |
| `verification_status=UNVERIFIED` | No independently reviewed ground-truth label is recorded. |
| `recording_status=VERIFIED` and `verification_status=VERIFIED` | A reviewer ID and explicit ground-truth label have been recorded for an accepted sample. |
| `quality_status=PASSED` | Technical array/preprocessing checks passed; this alone does not establish label correctness. |

Accepting a sample is not the same as verifying it. The review flow can inspect
the sequence and its technical quality, but the target remains a prompt until
the performed sign is independently verified.

### 4.4 Integrity and provenance checks

The current self-recording path includes these safeguards:

- checks raw landmark array dimensions, minimum frame count, missing-value
  structure, and finite processed values;
- requires at least eight raw frames and hand landmarks in at least three
  frames; low hand visibility can produce a warning;
- recomputes preprocessing and compares saved positions, presence flags, and
  model features against the current preprocessing functions;
- stores the sequence in an NPZ archive without requiring object/pickle loading
  in the sequence reader;
- calculates a SHA-256 from the raw landmarks and exact model features, and
  checks it against the SQLite record;
- checks that artifact paths remain inside self-recorded storage;
- can compare NPZ metadata with SQLite metadata before export;
- prevents duplicate source IDs/sequences across self-recorded splits and
  prevents one signer from being assigned to multiple dataset splits;
- validates feature shapes, labels, class-index mapping, vocabulary, source
  subset membership, and signer separation before training.

This catches missing, malformed, mismatched, or duplicate self-recorded
artifacts. It is not a visual near-duplicate scan of the ASL Citizen videos:
the current audit did not perform a content-level duplicate/near-duplicate
search on that source set.

### 4.5 Training eligibility gate

> **A recorded or accepted sample is not automatically training data.**

| Requirement | Why it is required |
|---|---|
| Accepted sample | A recording must pass the human review decision. |
| Explicit verified ground truth | Training labels must describe the sign actually performed, not only the target prompt or model output. |
| `quality_status=PASSED` and valid sequence | The arrays and preprocessing must be usable and consistent. |
| Matching sequence hash and metadata | Detects mismatches between the NPZ artifact and SQLite record. |
| Model-compatible feature dimensions and vocabulary mapping | The checkpoint expects the configured feature and class-index shape. |
| Non-empty signer ID and session ID | Preserves participant/session provenance. |
| Explicit `train`, `validation`, or `test` assignment | Prevents unassigned recordings from entering model data. |
| Signer-disjoint splits and no duplicate sample/sequence across splits | Reduces participant leakage and accidental reuse. |

The trainer only integrates self-recorded data when
`--self-recorded-features-dir` is explicitly supplied. It rejects unverified
archives, mismatched split metadata, incomplete provenance, duplicate IDs or
hashes, and a self-recorded feature directory with no training samples. The
baseline checkpoint path is protected from use as the candidate output.

The current 30-record self-recording database has **zero eligible samples**:
all accepted samples lack verified ground truth, and none is assigned to a
dataset split. Some legacy accepted rows also lack verification/quality
metadata. They cannot be treated as verified training data merely because their
database status says `accepted`.

## 5. Landmarks and feature processing

### 5.1 Landmark extraction

`src/extract.py` uses the MediaPipe Holistic Landmarker in video mode. It
selects **98 points per frame**:

- 13 upper-body pose points, including the shoulders, elbows, wrists, and
  selected face-adjacent pose points;
- 21 points for each hand (42 total);
- 43 selected face points around the mouth, brows, eyes, nose, and face outline.

Each point has normalized image `x`, `y`, and `z` coordinates. A point not
detected is represented by NaN in the raw landmark array, not by zero (zero can
be a valid coordinate). A saved landmark file represents a sequence with shape
`[frames, 98, 3]`.

### 5.2 Preprocessing shared by training and inference

The current `preprocess_sequence` flow is:

```text
Raw landmarks [T, 98, 3], missing points as NaN
      → derive left/right hand-presence flags
      → trim leading/trailing frames without a detected hand, keeping a margin
      → interpolate missing coordinates over time where possible
      → shoulder-center and shoulder-width normalization
      → linearly resample to 64 frames
      → replace any remaining NaNs with zero
      → build position, velocity, and hand-presence features
```

Normalization uses the median shoulder midpoint as the origin and median
shoulder width in the image plane as the scale. When the shoulders are entirely
unavailable, the fallback is a zero center and scale of 1; an invalid or
near-zero measured shoulder width also falls back to 1. This keeps preprocessing
defined, but missing shoulders can still reduce useful spatial information.

“Raw landmarks” are detector coordinates with possible missing values.
“Normalized positions” are body-relative coordinates after preprocessing.
Neither is the same as the final model features:

```text
positions:      98 points × 3 coordinates = 294 values/frame
velocities:     frame-to-frame differences = 294 values/frame
hand presence:  left/right flags           =   2 values/frame
                                             ───
model input:                                 590 values/frame
```

The final input to the classifier is therefore `[64, 590]` per sample. Training
augmentation (time warp, small rotation/scale/translation, and noise) is
applied to the training dataset only.

## 6. Sequence model

A sign is not just a still image. Its trajectory can matter:

```text
Frame 1 → Frame 2 → Frame 3 → … → Frame 64
```

The saved `models/gru_seed3.pt` checkpoint is a 50-class, bidirectional GRU
sequence classifier. Its network projects each 590-value frame to a hidden
representation, processes the ordered sequence in both directions with a
two-layer GRU, pools the temporal outputs using both mean and maximum, and maps
the pooled vector to class logits. Softmax converts those logits into scores
that sum to one.

A GRU is a recurrent neural network with learned gates that carry useful
information between time steps. Bidirectional processing reads the fixed
sequence forward and backward. This helps the classifier learn motion patterns
within a clip; it does not make the model understand ASL grammar or language.

The model learns associations from labeled motion sequences. The model
architecture also supports an LSTM option for training, but the current
`gru_seed3.pt` checkpoint is the baseline used by the desktop application.

The repository currently contains four GRU checkpoints (`gru_seed0.pt` through
`gru_seed3.pt`) and three LSTM checkpoints (`lstm_seed0.pt` through
`lstm_seed2.pt`). **No new candidate checkpoint is present in `models/` as of
the status snapshot below.** Keep `gru_seed3.pt` as the reproducible baseline;
write future experiments to a separate candidate path and compare them before
adopting one.

## 7. Real-time recognition

The unified desktop application starts in Demo mode and loads a checkpoint
(default: `models/gru_seed3.pt`). Its recognition worker:

1. reads webcam frames and runs MediaPipe on frames used for recognition;
2. appends the resulting 98-point arrays to a rolling buffer of 64 frames;
3. once the buffer is full, applies the same preprocessing and feature builder
   used by training;
4. runs inference every other processed frame in the unified application;
5. converts logits to probabilities and ranks canonical labels;
6. passes the best candidate, confidence, and hand-presence signal to the
   subtitle stabilizer;
7. displays candidate information and, after confirmation, appends the word to
   the subtitle history.

```text
Webcam frame → MediaPipe → 64-frame window → 590-feature sequence
             → model probabilities → top candidate
             → stabilization → committed subtitle word
```

The initial rolling window can contain idle/pre-sign frames, and subsequent
windows overlap. Offline evaluation processes a whole isolated clip after
trimming and resampling it. The impact of this live-window/whole-clip difference
has not been measured on a verified saved webcam evaluation set.

The app's confidence threshold defaults to `0.45`. A low-scoring candidate may
still be shown as a current prediction, but it does not qualify as a committed
subtitle word. The displayed score is the model's softmax score, not a
guarantee of correctness.

## 8. Prediction stabilization

Frame-by-frame output can flicker:

```text
HELLO → THANK YOU → HELLO → HELLO → THANK YOU
```

The system distinguishes a **raw/current prediction** from a
**committed subtitle word**. `SubtitleManager` uses a state machine
(`IDLE → CANDIDATE → COOLDOWN`), a 10-update vote window, and a default
confidence threshold of `0.45`.

- Normally it requires at least five votes for a candidate.
- It can confirm after three votes only when all relevant candidate votes have
  confidence of at least `0.90`.
- Low-confidence predictions or missing hands are invalid for confirmation.
- After a word is committed, it waits at least 0.8 seconds and requires eight
  consecutive invalid updates before re-arming.
- A continuing sign therefore does not immediately append the same word on
  every window. It can be committed again after the state machine re-arms.

These settings are configurable through the application arguments. Stabilizing
the output makes the UI more readable; it does not make an incorrect model
prediction correct.

## 9. Subtitle generation

The subtitle layer appends **word-level canonical labels in chronological
order**. It is a post-processing/display layer, not a natural-language
translator. It does not add grammar, punctuation, or sentence meaning.

The label display code uses explicit canonicalization rules. For example,
`WHATFOR1`, `WHATFOR2`, and `WHATFOR3` are grouped as `WHAT FOR` because the
source metadata identifies them as the same sign. The original model label and
class index are still preserved. Other configured variants remain distinct
(for example, `CHRISTMAS1` and `CHRISTMAS2`); numeric suffixes are not stripped
indiscriminately.

The subtitle history:

- receives words only after stabilization;
- retains order instead of replacing the previous word with the latest label;
- holds at most 50 words and supports clearing/resetting the history;
- suppresses repeated commits while a sign remains active through the
  cooldown/re-arm behavior.

The desktop UI and the `src/realtime.py` compatibility entry point launch the
same unified application. The older `run()` implementation in
`src/realtime.py` has its own legacy loop; the module's command-line entry
delegates to `application.main`.

## 10. Testing Mode

Testing Mode is for structured prompted attempts, not casual Demo-mode use. It
can select a vocabulary target, optionally play a catalog reference, capture a
webcam attempt, and store the model result and context in the attempts
database. When a stable prediction is confirmed, the system can save the exact
raw landmark window, normalized positions, hand flags, and model features with
checkpoint/preprocessing metadata for replay.

The attempt record separates:

- prompted target;
- top predictions and confidence;
- independently reviewed ground-truth label and correctness status;
- attempt status such as completed, no prediction, invalid, cancelled,
  interrupted, or error.

For example:

```text
Target prompt:         RIVER1
Model prediction:      WHAT FOR
Verified ground truth: not yet recorded
Evaluation result:     unverified (not counted as correct or incorrect)
```

A target/prediction match can be shown as target agreement, but it is not
ground-truth accuracy until a reviewer verifies what was actually performed.
Testing Mode does not retain the webcam video for an attempt; it may retain
the stabilized model-input sequence. Retrospective review therefore requires
an independent observer or external evidence.

Reference videos are optional. The current manifest contains 50 vocabulary
entries but no verified video artifacts, so the current UI reports that demos
are unavailable rather than providing verified examples.

## 11. Evaluation method and current results

### 11.1 Accuracy

Top-1 accuracy is the number of test examples whose highest-ranked predicted
label matches the reference label, divided by the number of evaluated examples:

```text
Accuracy = Number of Correct Predictions / Total Number of Predictions
```

For example, 85 correct predictions among 100 examples gives `85 / 100 = 85%`.
Accuracy is easy to interpret as the overall fraction correct. It does not show
which signs are difficult, whether errors concentrate in a class, whether the
class distribution is imbalanced, or whether the test set represents the
intended deployment population. It should be reported on held-out data, not
data used to fit the model.

### 11.2 Baseline and candidate comparison

The training code supports evaluating the baseline checkpoint and trained
candidate on the same held-out test feature split:

```text
Same held-out test split
      ├─ baseline checkpoint → baseline metrics
      └─ candidate checkpoint → candidate metrics
```

Using the same examples makes the comparison more controlled. The test set
should remain reserved; use validation data for model/threshold decisions.
The current baseline is `models/gru_seed3.pt`. No new candidate was found in the
repository status snapshot, so **no candidate improvement is claimed**.

### 11.3 Class-level and confusion analysis

The training evaluation code reports top-1/top-5 accuracy, macro-F1,
per-class precision/recall/F1 and counts, and a confusion matrix. A class-level
view can reveal weak signs hidden by an overall score. For example, “90% overall
accuracy” could coexist with `HELLO: 98%`, `THANK_YOU: 95%`, and `PLEASE: 45%`;
those numbers are a conceptual illustration, not this project's results.

In a confusion matrix, the row/reference label is the actual labeled sign, and
the column/prediction is the model output. It can show frequently confused
signs, but does not by itself establish why the confusion occurs.

For the current baseline's ASL Citizen test set, the report lists common
confusions including `THEY1 → DOG1` and `AXE1 → NOON1` (five examples each).
These are source-test confusions, not a webcam error analysis.

Testing Mode has a separate report based on independently verified attempts.
It can report verified top-1/top-k, per-sign outcomes, confidence buckets,
coverage, common confusions, and a confusion matrix when at least five verified
examples are available. Per-sign rankability requires at least three examples.
Unverified attempts are excluded from verified accuracy.

### 11.4 Signer-separated evaluation

A random clip split can put the same signer in both training and test data:

```text
Weak split:         Signer A → train and test
Signer-separated:  Signers A/B → train; unseen Signer C → test
```

Signing style may be shared across clips from one person, so signer overlap can
make evaluation easier than testing on a new person. The active source subset
uses disjoint signer groups for train, validation, and test, and training
validates that separation. This reduces one leakage risk; it does not prove
generalization to all signers. The source content was not checked for
near-duplicate videos under different IDs.

### 11.5 Existing offline baseline results

The repository's `reports/gru_seed3_metrics.json` and
`reports/model_reality_audit.md` report the following for `gru_seed3.pt` on the
held-out ASL Citizen test feature archive:

| Measure | Result | Interpretation |
|---|---:|---|
| Test top-1 accuracy | 499 / 699 = **71.4%** | Overall top prediction correct on this source test set. |
| Test top-5 accuracy | 637 / 699 = **91.1%** | Correct label appears among five highest-ranked classes. |
| Test macro-F1 | **0.710** | F1 averaged equally across the 50 classes. |
| Best validation top-1 | 138 / 179 = **77.1%** | Best validation result recorded for the checkpoint. |
| Coverage at confidence threshold 0.45 | 510 / 699 = **73.0%** | Fraction with a top score at or above the threshold. |
| Accuracy among covered examples | 444 / 510 = **87.1%** | Selective accuracy after excluding 189 lower-confidence examples. |

The 87.1% figure is **not** accuracy over all 699 examples; it only describes
the subset the threshold would answer. None of these figures is a real-world
webcam accuracy result. The test split has 11 held-out signers, and the report
shows substantial variation among them.

### 11.6 Current evaluation evidence

- A baseline checkpoint and offline test report exist.
- The repository contains 950/191/729 selected source examples in
  train/validation/test CSVs; feature generation retained 925/179/699.
- The local Testing Mode database currently has eight attempt records: four
  completed and four with no prediction. All eight are unverified. Four
  sequence archives are present, but none has verified ground truth.
- Therefore, there is **no verified real-webcam accuracy estimate** in the
  current records. Attempts may have been made, but unverified target prompts
  and predictions are not a measured evaluation.
- There is no candidate checkpoint in the models directory and no evidence of
  a newly trained candidate in the current artifacts.
- The full unittest suite passed: 86 tests in 14.382 seconds using Python
  3.14.5 (`python -m unittest discover -s tests -v` from `project/`).

## 12. Improvements in the current implementation

The current codebase contains the following safeguards and refinements compared
with a simple “latest frame → label” demo. This describes implemented behavior;
it does not assert that every change produced a measured accuracy improvement.

| Problem | What the implementation does | Why it matters |
|---|---|---|
| Predictions can flicker between labels. | Uses confidence gating, a vote window, repeated-vote confirmation, and cooldown/re-arm logic. | Avoids immediately committing every changing model output. |
| New predictions could overwrite prior output or repeat continuously. | Maintains an ordered subtitle history and waits for invalid/idle updates before re-arming. | Produces a more readable word stream while still allowing a later repeated sign. |
| A prompt does not prove the signer performed the target sign. | Separates target label, prediction, review decision, and independently verified ground truth. | Reduces the risk of treating a prompt or model guess as a training label. |
| Anonymous clips make leakage hard to detect. | Stores signer and session identifiers and rejects cross-split signer reuse for the self-recorded pipeline. | Supports more interpretable splits and traceability. |
| Stored sequences and metadata can drift apart. | Validates dimensions/preprocessing and checks sequence hashes and NPZ/SQLite metadata. | Prevents many malformed or mismatched samples from silently entering export. |
| New data could overwrite a baseline or change class mapping silently. | Protects `gru_seed3.pt` as a candidate output, validates vocabulary/index alignment, and supports a separate candidate artifact. | Preserves a reference model and makes experiments more comparable. |
| User-recorded data could be mixed into training implicitly. | Requires explicit feature-directory opt-in and verified, quality-passed samples assigned to controlled splits. | Makes training use deliberate rather than automatic. |
| Runtime preprocessing could diverge from training. | Training and inference share `preprocess_sequence` and `build_features`; the live hand-activity gate was aligned with the completeness rule used by preprocessing and has regression tests. | Reduces implementation drift. This correction has not been shown to improve webcam accuracy. |
| Testing results could be confused with truth. | Separates target agreement from verified accuracy and excludes unverified attempts from verified metrics. | Keeps prompted practice results from becoming unsupported evaluation claims. |

Still incomplete: more verified signers and samples are needed, all current
reference demos are missing, there is no verified webcam evaluation set, and no
new candidate has been trained/evaluated.

## 13. Current project status

Status snapshot: **2026-11-06**, based on the repository files and local data
artifacts inspected for this README.

| Component | Status |
|---|---|
| MediaPipe landmark extraction | Implemented; pretrained Holistic task artifact is in `project/models/`. |
| Shared normalization and feature construction | Implemented; 64 frames × 590 features. |
| GRU/LSTM training code | Implemented; active data archives and baseline checkpoints exist. |
| Webcam Demo mode | Implemented in the unified desktop app. |
| Prediction stabilization and subtitle history | Implemented. |
| Guided Testing Mode and SQLite attempt history | Implemented; attempts can remain unverified. |
| Exact stabilized Testing Mode sequence snapshots | Implemented for eligible completed attempts. |
| Self-recording Studio and sample review | Implemented; local sample database contains 30 records. |
| Independent ground-truth verification and split gate | Implemented; currently no self-recorded sample has passed the full training gate. |
| Verified reference demo videos | **Unavailable:** current catalog has 0 of 50 verified videos. |
| Baseline offline evaluation | Available for `gru_seed3.pt` on the ASL Citizen test split. |
| Candidate model retraining | Training code exists; no new candidate checkpoint is present. |
| Verified real-webcam evaluation | **Not available:** local attempts have no verified ground truth. |
| Large, diverse self-recorded dataset | **Not available:** current database has one signer/session and no verified samples. |
| Automated tests | 86 unit tests passed (Python 3.14.5). |

## 14. Current data inventory

The current active source subset has 50 labels, with signer counts of 31 in
training, 5 in validation, and 11 in test. The active feature archives contain
925 train, 179 validation, and 699 test samples after missing/low-hand-visibility
examples are filtered. The active subset metadata has no signer overlap between
the three splits.

The self-recorded database currently contains:

| Database status / fields | Count |
|---|---:|
| Total self-recorded sample rows | 30 |
| Pending review | 0 |
| Accepted | 29 |
| Rejected | 1 |
| Accepted but unverified, with quality recorded as passed | 24 |
| Legacy accepted-unverified rows with incomplete verification/quality metadata | 5 |
| Verified ground truth | 0 |
| Assigned train/validation/test/final-test/real-world split | 0 |
| Eligible self-recorded training samples | **0** |
| Distinct signer IDs / session IDs in this database | 1 / 1 |

The separate Testing Mode database has eight attempts, all unverified (four
completed, four no-prediction). This supports neither a verified test score nor
ground truth labels. The four saved Testing Mode sequences are stored as replay
artifacts, not as approved training data.

These counts are a snapshot of local repository data and may change as samples
are reviewed. Re-run the audit/management commands before making a later
presentation with updated counts.

## 15. How to run the project

### 15.1 Environment and dependencies

The application code and data live under `project/`; run application commands
from that directory. The repository-level `requirements.txt` currently lists
NumPy, pandas, matplotlib, scikit-learn, Pillow, and Jupyter. It does **not**
install the full webcam/training stack. The application also imports PyTorch,
MediaPipe, and OpenCV; data extraction uses `tqdm`; the optional subset
downloader uses `remotezip`. Compatible versions for the missing packages are
not pinned in the repository. Install platform-compatible packages for the
work you intend to run; in particular, choose the PyTorch build for your
machine. Tkinter is a Python distribution/platform component.

PowerShell setup from the repository root:

```powershell
cd .\project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ..\requirements.txt
```

Install the extra runtime/extraction dependencies described above separately
when needed. Do not assume the requirements file alone is enough to start the
webcam application.

### 15.2 Launch the application

From `project/`:

```powershell
python src\demo.py --ckpt models\gru_seed3.pt
```

The application opens in Demo mode. Use the app navigation to switch to
Testing Mode; open Self-recording Studio from the Testing page.
`src\realtime.py` also launches the unified application. Useful options include:

```powershell
python src\demo.py --ckpt models\gru_seed3.pt --mode testing --target AXE1
python src\demo.py --ckpt models\gru_seed3.pt --debug-performance
```

Testing Mode can optionally receive an anonymous signer ID and session notes
using `--signer-id` and `--session-notes`.

### 15.3 Run the automated tests

From `project/`, with the project dependencies installed:

```powershell
python -m unittest discover -s tests -v
```

The tests cover the application/worker, data stores, preprocessing/evaluation
helpers, label handling, and subtitle logic. The command was last verified in
the repository environment with 86 tests passing under Python 3.14.5; rerun it
after source changes and in the environment used for a presentation/demo.

### 15.4 Record and review samples

The recommended current path is the app's **Self-recording Studio**:

1. Select the target vocabulary item.
2. Start a session with an anonymous signer ID.
3. Record and inspect the preview and quality information.
4. Save as pending review or discard.
5. Review and accept/reject; accepted samples remain unverified.
6. Independently enter the performed sign and reviewer ID.
7. Assign an explicit split only after verification.

The management CLI can inspect and manage stored samples. Run these from
`project/`:

```powershell
python src\manage_recordings.py dashboard
python src\manage_recordings.py pending
python src\manage_recordings.py inspect <sample-id>
python src\manage_recordings.py review <sample-id> --decision accepted --notes "Reviewed"
python src\manage_recordings.py verify <sample-id> --ground-truth AXE1 --reviewer-id R001
python src\manage_recordings.py assign-split <sample-id> --split train
```

Replace placeholders with real sample IDs and vocabulary labels. A sample can
only be split after acceptance, technical validation, and explicit ground-truth
verification. Split assignment also enforces signer separation.

`src/record_self.py` is an older video-plus-CSV recorder. Its output is not
automatically imported into the current reviewed store; do not treat its
manifest's target or split columns as verified training provenance.

### 15.5 Prepare features and train a candidate

The existing baseline feature files are under `data/features/`. Candidate
training from the current baseline dataset can be started with:

```powershell
python src\train.py --kind gru --seed 0 --output models\gru_candidate.pt
```

This trains a new artifact; it has not been run as part of this README update.
The trainer validates source split shapes, vocabulary/label mappings, class
coverage, and signer separation; it evaluates the candidate and the configured
baseline checkpoint on the same test protocol. It will not overwrite
`gru_seed3.pt` as the candidate output. Existing output files require explicit
overwrite authorization and are backed up.

To explicitly integrate reviewed self-recorded data, first ensure it has
verified ground truth, passed technical checks, and signer-disjoint assignments.
Then prepare the separate archives and pass their directory explicitly:

```powershell
python src\manage_recordings.py prepare-features --output data\self_recorded\features
python src\train.py --kind gru --seed 0 --output models\gru_candidate.pt `
  --self-recorded-features-dir data\self_recorded\features
```

The current local store has no eligible samples, so this integration path is not
ready to provide useful self-recorded training data yet. A candidate still
needs evaluation and review before it should replace the baseline.

### 15.6 Optional source-data preparation and audit

The current data/features/models are already present in this repository. For a
fresh setup, the source-data preparation scripts are available; downloading
requires the optional `remotezip` dependency and the ASL Citizen archive:

```powershell
python src\download_model.py
python src\fetch_subset.py csv
python src\make_vocab.py --n 50 --pick top
python src\fetch_subset.py videos
python src\extract.py --workers 4
python src\dataset.py --min-hand-ratio 0.2
```

The video download and landmark extraction can require substantial time and
storage. The source subset is selected by `make_vocab.py`; do not rerun it
casually against existing artifacts because it rewrites the active vocabulary
and subset CSVs.

To refresh reports from the current active data:

```powershell
python src\validate_dataset.py --quick
python src\audit_dataset.py --checkpoint models\gru_seed3.pt
```

These commands write reports under `reports/`. `reports/audit.md` is marked as
a historical, superseded audit. `reports/model_reality_audit.md` is dated
2026-11-05; use it and the checkpoint metrics for the offline baseline results,
but use the current database inspection above for newer Testing Mode and
self-recording counts.

## 16. Presentation demo flow

1. **Introduce the task.** Explain isolated sign recognition and why video
   landmarks are processed as sequences rather than as one still image.
2. **Show Demo mode.** Run the webcam app and point out landmark visualization,
   top predictions, and confidence.
3. **Show stabilization.** Contrast the changing current prediction with a
   committed word in the subtitle history; explain the vote/cooldown stage.
4. **Show Testing Mode.** Select a target and explain that a prompt is not
   ground truth. Note that current reference videos are unavailable in the
   catalog and local attempts remain unverified.
5. **Show the data gate.** Walk through `record → review → verify → assign split
   → prepare features → train candidate`. Point out that the current
   self-recorded samples are not eligible.
6. **Explain the existing evaluation.** Show the held-out ASL Citizen baseline
   metrics and emphasize that they are not webcam results.
7. **Close with limitations.** Be explicit about signer/data coverage,
   continuous-signing limitations, missing verified webcam metrics, and the
   lack of a new candidate model.

## 17. Presentation talking points

### “What is the core contribution?”

The project demonstrates an end-to-end isolated-sign pipeline: landmark-based
temporal features, live sequence inference, post-processing for readable
word-level output, and a provenance-aware path for collecting and evaluating
new data. Do not present those engineering safeguards as a measured accuracy
improvement.

### “Why not use a normal image classifier?”

A single frame does not represent the motion across a sign. This project
classifies a fixed sequence of landmark-derived positions and velocities.

### “How do you know the model is accurate?”

The existing checkpoint has an offline result on a signer-separated ASL Citizen
test split: 499/699 top-1 correct (71.4%). That is evidence for that test set,
not a verified webcam result or universal ASL accuracy.

### “Why store signer IDs?”

Signer IDs make it possible to keep a person's examples together in one split
and reduce leakage from the same signing style appearing in both training and
test sets.

### “Why can't every recording be used for training?”

The target is only an instruction. The sign performed may differ, and a
technically valid sequence may still have the wrong label. The current gate
requires acceptance, verified ground truth, passed quality checks, matching
provenance, and a controlled split.

### “How does the subtitle system reduce flickering?”

It commits only candidates that pass the confidence rule and repeated-vote
stabilization. It then waits for cooldown and invalid/idle updates before
accepting another sign.

### “What happens when confidence is low?”

The current prediction may still be visible as a candidate, but it does not
count as a valid vote for subtitle commitment. If no stable prediction is
confirmed during a Testing Mode attempt, that attempt can be stored as
`no_prediction`.

### “What is the difference between the baseline and a candidate?”

`gru_seed3.pt` is the current comparison checkpoint. A candidate is a separately
trained artifact evaluated on the same held-out protocol. No new candidate or
improvement claim is present in the current repository snapshot.

## 18. Limitations

- The classifier has a fixed 50-sign vocabulary and recognizes isolated clips;
  it does not segment arbitrary continuous signing into translated sentences.
- Training and offline test data come from ASL Citizen, while webcam conditions
  can differ in camera, framing, lighting, signing speed, and signer style.
- The local self-recorded collection has one signer/session, zero verified
  ground-truth labels, and zero assigned split examples.
- Testing Mode has eight recorded attempts, but all are unverified; they cannot
  establish webcam accuracy.
- No verified sign demonstration videos are currently available in the catalog.
- Hand, pose, or shoulder landmarks can be missing. Fallback preprocessing
  keeps the sequence numerically processable but cannot restore missing visual
  information.
- The live sliding window can include idle or boundary frames, unlike
  whole-clip preprocessing in the offline test set. Its measured impact is
  unknown.
- Confidence is a softmax score; thresholding does not guarantee calibrated
  uncertainty or correctness.
- A source-split metadata audit found signer-disjoint splits, but a content
  hash/near-duplicate scan of the ASL Citizen source videos was not performed.
- Current offline performance varies by held-out signer and class; overall
  accuracy alone hides those differences.

## 19. Future work

These are next steps, not completed features or results:

- collect consented, technically valid recordings from more signers;
- have performed signs independently reviewed and keep provenance complete;
- reserve a signer-separated webcam evaluation set before using new recordings
  for training or threshold tuning;
- compare offline and live predictions on the same saved landmark sequences;
- populate and verify the reference demonstration catalog;
- examine per-sign and per-signer errors before deciding which classes or
  preprocessing choices need attention;
- train a separate candidate only after data checks, then compare it with the
  baseline on the same held-out evaluation protocol;
- investigate continuous-sign segmentation and sentence-level modeling as a
  distinct future research problem.