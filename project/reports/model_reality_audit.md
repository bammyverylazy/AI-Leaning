# Model Reality Audit

**Audit date:** 2026-11-05  
**Checkpoint:** `models/gru_seed3.pt`  
**Scope:** active dataset split, saved features and reports, checkpoint metadata, offline evaluation, live inference path, and available Testing Mode records.

## Executive decision

The model has **moderate source-dataset performance on an unseen-signer test split**, but there is not enough verified real-world data to claim webcam accuracy. The most concrete runtime defect found was a hand-activity gate that used a different hand-completeness rule from training preprocessing; that mismatch is fixed and covered by regression tests. A remaining structural mismatch is that training receives an isolated clip trimmed and resampled as a whole, while live inference receives overlapping 64-frame windows from continuous signing. Its effect has not been measured on a saved webcam sequence.

**Recommendation: Option 4, staged.** Keep the corrected runtime, collect a separate multi-signer webcam evaluation set, and evaluate it before using any of it for training. If performance is poor, expand the training set with curated, consented webcam data and retrain as a traceable experiment. Do not tune the threshold on the reserved evaluation set or overwrite the existing checkpoint. No retraining was performed in this audit.

## Current checkpoint and data

- The checkpoint contains a 50-class GRU, hidden size 192, dropout 0.3, a 590-feature input, and the 50-label vocabulary in class-index order. Each example is a 64-frame sequence. The feature archive stores 98 XYZ landmarks per frame, hand-presence features, and integer labels.
- The checkpoint vocabulary starts with `AXE1` and ends with `WHATFOR1`; it matches the active vocabulary. No class-index shift was found in the inspected artifacts.
- Active subset CSVs contain 950 train, 191 validation, and 729 test examples. Their split signer counts are 31, 5, and 11. Feature generation retained 925, 179, and 699 examples after its hand-visibility filter.
- Per-class subset counts are fairly balanced in training (18–24; median 19). Validation has 3–6 (median 4), and test has 9–20 (median 15). The small validation counts make the selected confidence threshold and per-class conclusions noisy.
- The source CSVs and active subset metadata have no repeated video IDs or signer overlap across train/validation/test. This supports a signer-independent split at the metadata level. Training augmentation is applied only to training items.
- No content-hash or visual near-duplicate scan was performed. Distinct filenames alone cannot rule out duplicate or near-duplicate video content. Label meaning was not independently re-annotated.

### Active subset counts by label

These are source-video counts in the active subset CSVs, before the feature-generation visibility filter.

| Label | Train | Val | Test |
|---|---:|---:|---:|
| AXE1 | 19 | 4 | 15 |
| BACKPACK1 | 19 | 3 | 15 |
| BASKETBALL1 | 19 | 5 | 20 |
| BEE1 | 19 | 6 | 14 |
| BELT1 | 19 | 3 | 18 |
| BITE1 | 21 | 4 | 14 |
| BREAKFAST1 | 21 | 3 | 15 |
| CANCER1 | 18 | 4 | 15 |
| CHRISTMAS1 | 19 | 3 | 17 |
| CONFUSED1 | 18 | 4 | 15 |
| DARK1 | 21 | 3 | 15 |
| DEAF1 | 20 | 4 | 15 |
| DECIDE1 | 20 | 3 | 15 |
| DEMAND1 | 21 | 4 | 14 |
| DEVELOP1 | 19 | 3 | 14 |
| DINNER1 | 18 | 4 | 15 |
| DOG1 | 24 | 4 | 17 |
| DOWNSIZE1 | 18 | 4 | 16 |
| DRAG1 | 18 | 3 | 17 |
| EAT1 | 18 | 4 | 17 |
| EDIT1 | 19 | 4 | 15 |
| ELEVATOR1 | 19 | 5 | 15 |
| FINE1 | 19 | 4 | 16 |
| FLOAT1 | 18 | 3 | 10 |
| FOREIGNER1 | 18 | 4 | 17 |
| GUESS1 | 18 | 5 | 14 |
| HALLOWEEN1 | 19 | 4 | 14 |
| HOSPITAL1 | 18 | 4 | 17 |
| HURDLE/TRIP1 | 22 | 3 | 13 |
| JEWELRY | 18 | 3 | 10 |
| KNIGHT1 | 18 | 3 | 10 |
| LOCK1 | 18 | 5 | 14 |
| LUNCH1 | 19 | 5 | 14 |
| MAPLE | 18 | 3 | 10 |
| MEAT1 | 18 | 3 | 10 |
| MECHANIC1 | 20 | 3 | 16 |
| MICROSCOPE1 | 18 | 4 | 16 |
| MOVIE1 | 18 | 4 | 17 |
| NOON1 | 19 | 4 | 15 |
| PARTY1 | 20 | 4 | 15 |
| PATIENT2 | 19 | 4 | 16 |
| RAZOR2 | 18 | 3 | 9 |
| RIVER1 | 19 | 4 | 15 |
| ROCKINGCHAIR1 | 20 | 4 | 15 |
| SHAVE1 | 19 | 5 | 15 |
| SINK | 18 | 4 | 10 |
| SQUEEZE | 18 | 3 | 10 |
| THEY1 | 18 | 3 | 15 |
| TWINS1 | 18 | 4 | 15 |
| WHATFOR1 | 20 | 5 | 18 |

## Offline evaluation

I re-evaluated the existing checkpoint directly on the active test feature archive through the existing evaluation callable, without training or writing a checkpoint. The resulting counts and scores match `gru_seed3_metrics.json` to floating-point rounding; the top-1 score is 499 correct predictions out of 699 examples. The confidence-filtered score is higher, but applies only to answered examples.

| Metric | Result |
|---|---:|
| Validation best top-1 | 138 / 179 (77.1%) |
| Test top-1 | 499 / 699 (71.4%) |
| Test top-5 | 637 / 699 (91.1%) |
| Test macro-F1 | 0.710 (699 test examples) |
| Test coverage at threshold 0.45 | 510 / 699 (73.0%) |
| Accuracy among answered examples at threshold 0.45 | 444 / 510 (87.1%) |

The 87.1% selective accuracy is **not** the model's accuracy over all test clips: it excludes 189 of 699 examples. The checkpoint metrics and active application both use threshold 0.45, so a threshold mismatch is not supported as the cause for this checkpoint. The threshold is not embedded in the checkpoint, however, so future runs should continue to record it with their evaluation.

Test accuracy varies substantially across the 11 held-out participants:

| Participant | Test examples | Accuracy |
|---|---:|---:|
| P15 | 50 | 86.0% |
| P17 | 53 | 50.9% |
| P18 | 43 | 72.1% |
| P22 | 50 | 78.0% |
| P35 | 99 | 75.8% |
| P42 | 95 | 74.7% |
| P47 | 53 | 90.6% |
| P48 | 54 | 87.0% |
| P49 | 50 | 78.0% |
| P6 | 96 | 47.9% |
| P9 | 56 | 58.9% |

Frequent source-test confusions include `THEY1 → DOG1` (5), `AXE1 → NOON1` (5), `LOCK1 → BITE1` (4), `HOSPITAL1 → FINE1` (4), and `DEVELOP1 → DEMAND1` (4). These are source-dataset confusions, not a substitute for webcam error analysis.

The lowest source-test class recalls were `CANCER1` (6/15, 40.0%), `DINNER1` (6/15, 40.0%), `THEY1` (5/12, 41.7%), `LUNCH1` (6/14, 42.9%), and `ELEVATOR1` (7/15, 46.7%). No class was never predicted. `EAT1` was the most frequent prediction (23 of 699), followed by `WHATFOR1`, `DOG1`, and `BITE1` (21 each); this is a source-test output distribution, not evidence of the same bias in webcam use.

## Cause analysis

| Candidate cause | Finding |
|---|---|
| Signer/data split leakage | **Not found in metadata.** Official split signers and video IDs are disjoint. Content-level duplicate videos under different IDs were not checked. |
| Severe class imbalance | **Not supported.** The active subset is fairly balanced; the validation split is small. |
| Incorrect class-index mapping | **Not found.** Active feature labels, vocabulary order, and checkpoint vocabulary align in the inspected artifacts. |
| Training/inference feature implementation mismatch | **Mostly ruled out by code reuse and a passing parity test.** Both paths call the same landmark ordering, normalization, sequence preprocessing, and feature builder. The fixed-window test confirms identical `[1, 64, 590]` float32 model input for the runtime and offline paths. |
| Runtime hand-activity gate | **Confirmed mismatch and fixed.** Training presence flags require all 21 x-coordinates of a hand. The app previously treated any detected x-coordinate as hand activity. The live gate now uses the same completeness rule; this reduces false activity detection but does not prove a webcam accuracy gain. |
| Real-time window/sequence mismatch | **Structurally present; impact unmeasured.** Offline clips are trimmed and resampled as whole isolated signs. Live recognition repeatedly evaluates overlapping 64-frame windows from continuous webcam input, which may include pre-sign or post-sign idle frames. There is no saved, labeled webcam sequence to compare end-to-end. |
| Domain shift | **Plausible and likely relevant, not yet quantified.** Training and test data are ASL Citizen recordings; the user's webcam environment, framing, lighting, signing speed, and signer style differ. No independent real-world evaluation set exists. |
| Overfitting | **Inconclusive.** The saved curves show train and validation loss both decreasing, with a modest final gap and validation accuracy around 77%. Training accuracy and complete per-epoch logs are unavailable; the available evidence does not justify a definitive overfitting claim. |
| Confidence threshold | **Not supported as the primary cause.** The active default 0.45 matches the threshold in the seed-3 report. Changing it cannot fix a wrong top class and should not be used to mask poor predictions. |

The code path uses shared preprocessing but does not have a prior direct test comparing a captured model input to the offline result for the exact same fixed landmark sequence. Reusing the same function removes implementation drift for an identical input; it does not remove differences in camera landmarks or temporal alignment.

## Real-world evidence and limitations

- No labeled files are present in a dedicated `real_world_eval` dataset.
- The local Testing Mode database contains one completed attempt: requested target `WHATFOR1`, model prediction `AXE1` at 54.3% confidence. Its ground truth is unverified and the landmark sequence was not saved. It is a useful diagnostic observation, **not** a verified accuracy sample.
- I did not perform a webcam session during this audit. Therefore no real-world accuracy percentage, exact webcam-vs-offline replay comparison, or causal attribution to domain shift can be reported.
- Existing `record_self.py` records raw clips but does not by itself establish independent ground truth or keep a reserved real-world evaluation set separate from training. The current self-recording manifest contains only its header.
- The old `reports/audit.md` and `reports/dataset_report.*` referred to the archived `old_vocab` subset. `dataset_report.json` and `.txt` were regenerated from the active subset using `validate_dataset.py --quick`; the historical audit is marked as superseded.

## Decision and next steps

1. Keep the 2026-11-05 runtime hand-gate fix. Do not alter the confidence threshold to improve appearance.
2. Record a small **evaluation-only** set using the actual deployment webcam and several signers. Start with the frequent source-test confusion pairs listed above, then cover the full vocabulary as feasible. Have a qualified independent reviewer verify each performed sign.
3. Keep this evaluation data out of training and model/threshold tuning. Preserve signer IDs (anonymous IDs are sufficient), target, verified performed label, recording conditions, and the exact landmark sequence or raw video needed for replay.
4. Compare the exact saved landmark sequence through offline and live feature construction. If those predictions differ, fix the deployment path. If both are wrong, that is evidence for domain/sequence mismatch or model/data limitations.
5. If the held-out webcam evaluation remains poor, expand with consented, reviewed recordings from multiple signers in the deployment setup, and split by signer. Run traceable experiments that change one factor at a time; retain the current checkpoint and reserved evaluation set.
6. There is currently no evidence-based reason to remove a class. Revisit individual classes after verified real-world confusion counts exist.

For planning a full 50-label collection, these are proposed scale targets, not measured requirements: **minimum viable** is 5 evaluation-only signers × 3 independently recorded takes per label (750 clips); **better** is 10 evaluation signers × 5 takes (2,500 clips); **ideal** is 20 evaluation signers × 10 takes (10,000 clips). Keep all evaluation signers out of training. If resources are limited, begin with a clearly labeled pilot emphasizing the source-test low-recall/confusion signs, then collect a separate sealed evaluation set before making final performance claims. Independently review sign meanings and clip quality; repetitions by one signer are not substitutes for signer diversity.

To refresh the active dataset summary from the project directory on Windows:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python src\validate_dataset.py --quick
```

The current offline metrics are a source-dataset baseline only. They are not a claim of real-world webcam accuracy.
