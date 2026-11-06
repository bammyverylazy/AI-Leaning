> **Historical report — superseded by [`model_reality_audit.md`](./model_reality_audit.md), 2026-11-05.** This report describes an older 50-sign vocabulary and is not an evaluation of the active `gru_seed3.pt` checkpoint. Use the current report and refreshed `dataset_report.json` instead.

# Project audit

## 1. Dataset size and samples per class
classes: 50   videos: train 775, val 177, test 589, total 1541
signers:  train 32, val 5, test 11
signer overlap train/val/test: {'train-val': 0, 'train-test': 0, 'val-test': 0}

per-class summary (videos per class):
      train  val  test  total
min    15.0  3.0   9.0   29.0
25%    15.0  3.0  11.0   30.0
50%    15.0  3.5  12.0   31.0
75%    16.0  4.0  12.0   31.0
max    19.0  5.0  14.0   38.0
mean   15.5  3.5  11.8   30.8

per-class counts:
             train  val  test  total
gloss                               
ALLOFSUDDEN     17    3    12     32
BANDAGE         17    4    10     31
BLANKET         15    4    11     30
BOW1            15    4    12     31
CAMERA          15    4    12     31
CAN1            15    4    12     31
CANNON          15    3    12     30
CAPTURE         17    3    12     32
COMMENT         15    4    12     31
COMMITTEE       15    3    13     31
COOL2           15    4    11     30
EACH            16    4    12     32
ENGINEER        15    4    12     31
FLAG2           15    5     9     29
FOLLOW4         15    4    11     30
FOOTBALL        15    3    12     30
GIRL            16    3    11     30
GYM             15    3    12     30
HANDSOME2       15    4    12     31
HEAD            16    4    11     31
HOLY            15    3    12     30
INTUITIVE       17    4    10     31
JEALOUS         15    3    13     31
LIGHT           15    3    13     31
LUNCH1          19    5    14     38
MAPLE           18    3    10     31
MIX2            15    4    11     30
NAVY            15    3    12     30
NO              16    3    11     30
PARAGRAPH       15    4    12     31
PEACE           15    4    12     31
PET1            16    3    12     31
POOR            15    3    11     29
PRECIOUS        16    3    12     31
QTIP            15    3    12     30
RELEASE         15    4    12     31
ROOF            15    3    12     30
RUN1            15    4    12     31
SAUSAGE         15    4    12     31
SHOOT2          15    3    13     31
SLOW            15    3    14     32
SOCCER1         15    3    12     30
SOUP            15    3    13     31
STRAW           15    4    11     30
TIEUP2          16    3    12     31
UGLY            15    3    12     30
VOICE           17    3    11     31
WATCH           15    4    12     31
WRITEONWALL     16    4    11     31
YEAR            15    4    12     31

## 3. Class imbalance and split balance
train: min 15  max 19  max/min 1.27  coefficient of variation 0.06
val: min 3  max 5  max/min 1.67  coefficient of variation 0.16
test: min 9  max 14  max/min 1.56  coefficient of variation 0.08
classes with < 3 val videos: 0   with < 8 test videos: 0
videos per signer (min / median / max):
     train   val  test
min    1.0   6.0  48.0
50%   13.0  51.0  54.0
max   57.0  52.0  57.0
chance accuracy = 0.020; majority-class baseline on test = 0.024

## Videos lost between subset and features
train: subset 775  landmark files 775  kept in features 753
val: subset 177  landmark files 177  kept in features 167
test: subset 589  landmark files 589  kept in features 577

## Landmark quality (_quality.csv)
status counts: {'ok': 1541}
hand_ratio percentiles (share of frames with any hand):
count    1541.000
mean        0.436
std         0.131
min         0.000
5%          0.233
10%         0.286
25%         0.359
50%         0.434
75%         0.512
max         1.000
frames per video percentiles:
count    1541.0
mean       82.4
std        36.3
min        19.0
5%         42.0
50%        75.0
95%       156.0
max       291.0
videos with hand_ratio < 0.2 (dropped by dataset.py): 44 of 1541

lowest mean hand_ratio by class (landmark-fragile signs):
gloss
INTUITIVE      0.300
COOL2          0.357
CAN1           0.365
ALLOFSUDDEN    0.382
GIRL           0.383
CAPTURE        0.391
ROOF           0.394
PEACE          0.396

mean hand_ratio by split:
split
test     0.443
train    0.432
val      0.430

## 2. Model performance (reports/)

gru_seed0_metrics.json: {"top1": 0.7487001733102253, "top5": 0.941074550151825, "macro_f1": 0.7494855275971378, "best_val_acc": 0.85628741979599, "conf_threshold": 0.5499999999999999, "test_coverage_at_thr": 0.6291161178509532, "test_acc_on_kept": 0.9008264462809917}
  per-signer test acc: min 0.45  median 0.81  max 0.94  (n signers 11)
  top confusions:
   count      true predicted
0      6      QTIP   JEALOUS
1      5      UGLY   JEALOUS
2      4     WATCH     VOICE
3      4  PRECIOUS     VOICE
4      4     MAPLE        NO
5      4    CANNON      POOR
6      3  PRECIOUS     LIGHT
7      3      PET1   FOLLOW4

lstm_seed0_metrics.json: {"top1": 0.7279029462738301, "top5": 0.927209734916687, "macro_f1": 0.7260533894471771, "best_val_acc": 0.826347291469574, "conf_threshold": 0.6, "test_coverage_at_thr": 0.608318890814558, "test_acc_on_kept": 0.8974358974358975}
  per-signer test acc: min 0.47  median 0.80  max 0.90  (n signers 11)
  top confusions:
   count         true predicted
0      5           NO      GIRL
1      3        WATCH     VOICE
2      3         UGLY   JEALOUS
3      3         QTIP      GIRL
4      3    COMMITTEE     COOL2
5      3  ALLOFSUDDEN   JEALOUS
6      2        WATCH     LIGHT
7      2       TIEUP2   COMMENT

## Are your demo words in the dataset / vocabulary?
HELLO: in current vocab = False; dataset matches (videos): HELLO=32
THANKYOU: in current vocab = False; dataset matches (videos): THANKYOU=31
GOODBYE: in current vocab = False; dataset matches (videos): none
BYE: in current vocab = False; dataset matches (videos): BYE=30
PLEASE: in current vocab = False; dataset matches (videos): PLEASE=31
SORRY: in current vocab = False; dataset matches (videos): SORRY=30
YES: in current vocab = False; dataset matches (videos): ORANGEEYES=31, CLOSEEYES=31, EYESAWAKE=31, EYES=30
NO: in current vocab = True; dataset matches (videos): NOON1=38, NORTH=32, NOSE=31, NOSEBLEED=31
HELP: in current vocab = False; dataset matches (videos): HELP=31
WATER: in current vocab = False; dataset matches (videos): WATERDROP=31, WATERSKI=31, WATER=30
EAT: in current vocab = False; dataset matches (videos): EAT1=39, EGGBEATER=32, MEAT1=31, SWEATER2=31
LOVE: in current vocab = False; dataset matches (videos): GLOVES2=31, GLOVES4=31, ALLOVER=31, GLOVES1=31