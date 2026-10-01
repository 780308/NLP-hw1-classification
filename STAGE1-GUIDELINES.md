# STAGE I — CLASSIFIER-AGNOSTIC HANDCRAFTED REPRESENTATION EXPERIMENT SPECIFICATION

## 0. Mission

Current official baseline:

- experiment: `DNN-001`
- input: RGB `64x64`, direct non-aspect-preserving resize
- network: `12288 -> 256 -> 64 -> 2`
- optimizer: AdamW
- test accuracy already observed: `62.60%`
- internal validation accuracy: `69.50%`
- DO NOT modify or overwrite `DNN-001`
- DO NOT reuse `data/val` during Stage I

Stage I goal:

1. determine whether non-neural handcrafted image representations improve cat/dog separability;
2. keep the classifier architecture fixed while comparing representations;
3. produce reusable spatial representations that can later feed:
   - DNN: flatten `[C,H,W] -> [C*H*W]`;
   - CNN: consume `[C,H,W]` directly;
   - RNN: convert to spatial sequence;
4. preserve all experiment provenance needed for the final report;
5. avoid neural-network-based or transformer-based feature extraction;
6. prioritize validation accuracy first, representation efficiency second;
7. do not perform blind hyperparameter searches.

Important assignment constraint:

- required raw-image DNN/CNN/RNN experiments remain valid independent baselines;
- Stage I representations are enhanced/extended inputs;
- do NOT silently replace the required raw-input CNN/RNN baselines with handcrafted features later.

---

# 1. Non-negotiable experimental rules

## 1.1 Data isolation

Use:

- `data/train`: all development, representation selection, model selection;
- `data/val`: untouched held-out final test.

During Stage I:

```text
data/val MUST NOT be:
- evaluated,
- inspected for prediction failures,
- used for threshold selection,
- used for normalization,
- used for feature selection,
- used for PCA,
- used for KMeans,
- used for any representation decision.
```

All Stage I decisions must use only internal splits of `data/train`.

Primary screening split:

```text
seed = 42
validation_fraction = 0.10
train = 1800
internal_validation = 200
```

Reuse exactly the same seed-42 split as `DNN-001`.

Confirmation seeds:

```text
42
123
2026
```

For seeds 123 and 2026 create new stratified 90/10 splits.

---

# 2. Core design decision: preserve a spatial feature map

Do NOT make the main Stage I artifact a single global BoVW/Fisher/HOG vector.

The canonical Stage I representation must have shape:

```text
[C, 7, 7]
```

where each spatial location corresponds to the same local region of the normalized image.

Reasons:

```text
DNN:
[C,7,7] -> flatten -> MLP

CNN:
[C,7,7] -> convolutional network

RNN option A:
[C,7,7] -> [7, 7*C]
one timestep per feature-map row

RNN option B, enhanced experiment only:
[C,7,7] -> [49, C]
one timestep per spatial cell
```

Prefer RNN option A when keeping similarity with the assignment's row-sequence formulation.

The Stage I representation must therefore retain local spatial structure.

---

# 3. Canonical image preprocessing

Implement one deterministic preprocessing function.

Create:

```text
representations/preprocess.py
```

Function concept:

```python
preprocess_for_handcrafted(image, output_size=128) -> np.ndarray
```

Required behavior:

1. apply EXIF orientation correction;
2. convert to RGB;
3. preserve original aspect ratio;
4. resize so the entire image fits inside `128x128`;
5. use high-quality downsampling:
   - PIL LANCZOS or OpenCV equivalent;
6. pad symmetrically to exactly `128x128`;
7. use reflection padding rather than constant black padding;
8. return deterministic RGB uint8 or well-defined float representation.

Required padding mode:

```text
reflect / BORDER_REFLECT_101
```

Do NOT:

```text
- crop the image;
- stretch arbitrary aspect ratios to 128x128;
- use random transforms;
- use neural object detection;
- use neural segmentation;
- use pretrained neural embeddings.
```

Reason for reflection padding:

- avoids the strong artificial edges introduced by black borders;
- preserves aspect ratio;
- remains deterministic.

Before running formal experiments, generate a diagnostic contact sheet for approximately 16 images with extreme aspect ratios.

Save runtime diagnostic only under:

```text
outputs/stage1/diagnostics/
```

Do not treat diagnostic images as experiment results.

---

# 4. Canonical spatial grid

Use one spatial layout for all handcrafted local features.

Image:

```text
128 x 128
```

Define:

```text
base cell size = 16 x 16 pixels
local region / block = 32 x 32 pixels
stride = 16 pixels
```

This creates:

```text
7 x 7 = 49 local regions
```

Top-left coordinates:

```text
x = 0, 16, 32, 48, 64, 80, 96
y = 0, 16, 32, 48, 64, 80, 96
```

Every local feature family must correspond to these same 49 spatial regions.

Do not silently change this grid for individual feature families.

---

# 5. Atomic feature families

Implement atomic feature extraction once.

Create:

```text
representations/extractors.py
```

All feature families must output:

```text
[C, 7, 7]
float32
```

## 5.1 HOG

Representation group name:

```text
hog
```

Use `skimage.feature.hog`.

Required configuration:

```python
orientations = 9
pixels_per_cell = (16, 16)
cells_per_block = (2, 2)
block_norm = "L2-Hys"
transform_sqrt = True
feature_vector = False
```

For `128x128` input this should produce conceptually:

```text
7 x 7 x 2 x 2 x 9
```

Reshape the block descriptor dimension:

```text
2 * 2 * 9 = 36
```

Canonical output:

```text
[36, 7, 7]
```

Do not manually reimplement HOG unless library behavior prevents the required output.

---

# 5.2 Uniform LBP

Representation group name:

```text
lbp
```

Use:

```python
P = 8
R = 1
method = "uniform"
```

Compute LBP on a grayscale uint8 image.

Do not call LBP directly on floating-point grayscale if the library warns against it.

Uniform LBP produces:

```text
P + 2 = 10 bins
```

For every canonical `32x32` local region:

1. collect its LBP values;
2. compute a 10-bin histogram;
3. normalize the histogram to sum to 1;
4. use zeros only if a region is unexpectedly empty.

Canonical output:

```text
[10, 7, 7]
```

---

# 5.3 Local HSV histogram

Representation group name:

```text
hsv
```

Purpose:

- supplementary local color information;
- NOT a primary global color descriptor because backgrounds contain unrelated objects.

Convert the normalized RGB image to HSV.

For every canonical `32x32` local region compute:

```text
Hue:
8 bins

Saturation:
4 bins

Value:
4 bins
```

Normalize each of the three histograms separately to sum to 1 before concatenation.

Per-region feature dimension:

```text
8 + 4 + 4 = 16
```

Canonical output:

```text
[16, 7, 7]
```

Do NOT use a large joint H-S-V histogram.

Do NOT use a single global HSV histogram as the primary representation.

---

# 5.4 Dense upright RootSIFT

Representation group name:

```text
rootsift
```

Use OpenCV SIFT from:

```text
opencv-python-headless
```

This is allowed because it is a traditional non-neural feature extractor.

Use 49 manually specified spatial keypoints aligned with the canonical `7x7` grid.

For each canonical `32x32` region:

```text
keypoint center:
x + 16
y + 16

keypoint size:
32

angle:
0 degrees
```

Use fixed upright orientation.

Do not run SIFT keypoint detection.

The locations must be deterministic.

Expected descriptor:

```text
128 dimensions per region
49 descriptors per image
```

Convert ordinary SIFT descriptor `d` to RootSIFT:

```python
d = d / (sum(abs(d)) + eps)
d = sqrt(d)
```

Use:

```python
eps = 1e-12
```

Validate:

```text
descriptor_count == 49
descriptor_dimension == 128
all finite
```

If OpenCV unexpectedly fails to return exactly 49 descriptors:

- do NOT silently reorder descriptors;
- stop the smoke test;
- inspect and fix keypoint construction.

Canonical output:

```text
[128, 7, 7]
```

Do NOT apply PCA in the initial screening phase.

Reason:

- preserve feature information;
- avoid introducing another learned component;
- avoid unsupervised validation leakage;
- RootSIFT map is already reasonably compact.

RootSIFT storage per image:

```text
128 * 7 * 7 = 6272 floats
```

This is already smaller than the current raw RGB flattened input:

```text
3 * 64 * 64 = 12288 floats
```

---

# 6. Master handcrafted feature bank

Do NOT repeatedly extract features separately for every representation experiment.

Extract the four atomic groups once from all 2000 images in `data/train`.

Canonical channel order:

```text
0:36     HOG       36 channels
36:46    LBP       10 channels
46:62    HSV       16 channels
62:190   RootSIFT 128 channels
```

Master tensor:

```text
[N, 190, 7, 7]
```

For the full training dataset:

```text
N = 2000
```

dtype:

```text
float32
```

Approximate cache size is acceptable.

Save:

```text
outputs/stage1/cache/handcrafted128/
    features.npy
    labels.npy
    filenames.json
    manifest.json
```

Use `.npy`, not compressed `.npz`, so later datasets can load with:

```python
np.load(..., mmap_mode="r")
```

Do not commit `features.npy`.

---

# 7. Master cache manifest

`manifest.json` must contain at least:

```json
{
  "schema_version": 1,
  "source_dataset": "data/train",
  "image_count": 2000,
  "image_size": 128,
  "preprocessing": {
    "aspect_ratio_preserved": true,
    "padding": "reflect",
    "random_augmentation": false
  },
  "grid": {
    "spatial_height": 7,
    "spatial_width": 7,
    "region_size": 32,
    "stride": 16
  },
  "channel_groups": {
    "hog": [0, 36],
    "lbp": [36, 46],
    "hsv": [46, 62],
    "rootsift": [62, 190]
  },
  "dtype": "float32",
  "feature_shape": [2000, 190, 7, 7],
  "dependency_versions": {},
  "code_commit": "",
  "config_sha256": "",
  "filenames_sha256": "",
  "extraction_seconds": 0.0,
  "bytes_on_disk": 0
}
```

Populate actual values programmatically.

Do not fabricate missing values.

---

# 8. Representation registry

Create:

```text
representations/registry.py
```

Define explicit representations.

Initial candidates:

```text
REP-001-HOG
channels = hog
shape = [36,7,7]
flatten_dim = 1764

REP-002-HOG-LBP
channels = hog + lbp
shape = [46,7,7]
flatten_dim = 2254

REP-003-ROOTSIFT
channels = rootsift
shape = [128,7,7]
flatten_dim = 6272

REP-004-ROOTSIFT-LBP
channels = rootsift + lbp
shape = [138,7,7]
flatten_dim = 6762

REP-005-ROOTSIFT-LBP-HSV
channels = rootsift + lbp + hsv
shape = [154,7,7]
flatten_dim = 7546
```

Do NOT initially run:

```text
HOG + RootSIFT
HOG + RootSIFT + LBP
all 190 channels
```

because HOG and SIFT both encode gradient information and may be redundant.

Fusion is conditional; see decision rules later.

---

# 9. Raw-image control

Keep `DNN-001` as the official assignment baseline.

Additionally implement one Stage-I control representation:

```text
REP-000-RAW64
```

Behavior:

```text
RGB
-> direct Resize((64,64))
-> ToTensor-style [0,1]
-> normalize mean=0.5,std=0.5
-> shape [3,64,64]
```

This reproduces the DNN-001 input representation, but Stage-I probe training will not use online horizontal flip.

Therefore:

```text
REP-000 is a Stage-I probe control.
DNN-001 remains the official historical baseline.
Do not claim REP-000 exactly reproduces DNN-001.
```

Also run one preprocessing control once:

```text
CTRL-LBOX64
```

Same DNN-001 model/training settings, but replace direct resize by:

```text
aspect-preserving resize + reflection padding to 64x64
```

Purpose:

```text
measure whether geometric distortion alone matters
```

This is a control experiment, not a candidate final representation.

---

# 10. Split-independent cache design

Because HOG/LBP/HSV/RootSIFT extraction is deterministic and contains no fitted parameters:

```text
extract the master feature bank once
```

Do NOT generate one feature cache per seed.

Splits must reference indices into the same master cache.

Create split files:

```text
report/splits/stage1_seed42.json
report/splits/stage1_seed123.json
report/splits/stage1_seed2026.json
```

Each split file contains:

```text
seed
validation_fraction
train filenames
validation filenames
class counts
sha256
```

Reuse the existing DNN-001 seed-42 split if identical.

Do not create a second different seed-42 split.

---

# 11. Feature normalization

Normalization is split-dependent.

Never calculate normalization statistics using internal validation data.

For each:

```text
representation
+
split seed
```

calculate per-channel statistics using only training indices.

For tensor:

```text
[N,C,H,W]
```

compute:

```text
mean[c] over N,H,W
std[c] over N,H,W
```

For any:

```text
std < 1e-6
```

use:

```text
std = 1
```

Normalize:

```text
x[c] = (x[c] - mean[c]) / std[c]
```

Save actual mean/std.

Do not overwrite raw master cache.

Runtime normalization should happen through a dataset/adapter.

This makes the same cached representation reusable for future DNN/CNN/RNN runs.

---

# 12. Stage-I probe classifiers

Feature comparison must use fixed classifiers.

Do not tune classifier architecture separately for each representation.

Every representation must be evaluated with BOTH:

```text
A. linear diagnostic probe
B. fixed MLP probe
```

---

# 12.1 Linear diagnostic probe

Purpose:

```text
measure how linearly separable the representation is
```

This is NOT the assignment's final classifier.

Use:

```text
sklearn LinearSVC
```

Configuration:

```python
C = 1.0
class_weight = None
dual = "auto"
max_iter = 10000
random_state = split_seed
```

Input:

```text
normalized feature map -> flatten
```

Do not tune `C` during initial screening.

Record:

```text
accuracy
cat_accuracy
dog_accuracy
balanced_accuracy
macro_f1
confusion_matrix
fit_seconds
```

---

# 12.2 Fixed MLP probe

Use exactly this architecture for all handcrafted representations:

```text
Flatten
Linear(D, 256)
ReLU
Dropout(0.3)

Linear(256, 64)
ReLU
Dropout(0.3)

Linear(64, 2)
```

Do NOT add BatchNorm in Stage I.

Do NOT change width/depth per representation.

Do NOT optimize dropout in Stage I.

Training configuration:

```text
loss = CrossEntropyLoss
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 32
epochs_max = 30
early_stopping_patience = 5
checkpoint selection = highest validation accuracy
tie break = lower validation loss
```

Keep this consistent with DNN-001 wherever possible.

Record model parameter count because input dimensionality changes.

Do not interpret reduced parameter count as an unfair disadvantage; compactness is part of representation quality.

---

# 13. Stage-I augmentation policy

Initial Stage-I screening:

```text
NO data augmentation
```

Reason:

```text
isolate representation quality
```

Do not precompute random crops, random rotations, random color jitter, MixUp, etc.

Horizontal-flip augmentation may be revisited only after the winning representation has been selected.

That belongs to Stage II or a clearly separated augmentation ablation.

---

# 14. Smoke tests before formal experiments

Run all of these before Phase A.

## 14.1 Preprocessing checks

Verify on diverse images:

```text
output shape == (128,128,3)
aspect ratio preserved
no unexpected cropping
reflection padding works
all images decode
```

## 14.2 Feature checks

For one image:

```text
HOG      == [36,7,7]
LBP      == [10,7,7]
HSV      == [16,7,7]
RootSIFT == [128,7,7]
master   == [190,7,7]
```

Verify:

```text
np.isfinite(...).all()
```

## 14.3 Cache checks

Verify:

```text
len(features) == len(labels) == len(filenames) == 2000
filenames unique
class count = 1000 cat / 1000 dog
filename-label mapping matches existing dataset loader
```

## 14.4 Tiny overfit test

Use one handcrafted representation and approximately 64 training examples.

Run the MLP probe with dropout disabled only for this diagnostic.

Target:

```text
training accuracy >= 95%
```

If it cannot substantially overfit 64 examples:

```text
stop;
debug feature ordering, labels, normalization, model, loss, optimizer.
```

Do not record the tiny overfit run as a formal result.

---

# 15. Phase A — single-split screening

Batch experiment ID:

```text
S1A-001
```

Split:

```text
seed = 42
```

Run:

```text
REP-000-RAW64
REP-001-HOG
REP-002-HOG-LBP
REP-003-ROOTSIFT
REP-004-ROOTSIFT-LBP
REP-005-ROOTSIFT-LBP-HSV
```

For every representation run:

```text
LinearSVC
fixed MLP probe
```

Do not access `data/val`.

Primary Phase-A ranking:

```text
MLP internal-validation accuracy
```

Secondary diagnostic metrics:

```text
LinearSVC validation accuracy
minimum(cat_accuracy, dog_accuracy)
macro F1
representation dimension
MLP parameter count
```

Do not rank primarily by training accuracy.

---

# 16. Phase-A interpretation rules

After all initial candidates finish, apply these exact rules.

## Rule 1 — LBP usefulness

Compare:

```text
REP-001 HOG
vs
REP-002 HOG+LBP
```

and:

```text
REP-003 RootSIFT
vs
REP-004 RootSIFT+LBP
```

If LBP decreases MLP validation accuracy by more than:

```text
1.0 percentage point
```

in BOTH branches:

```text
remove LBP from later fusion candidates
```

Otherwise retain it.

---

## Rule 2 — HSV usefulness

Compare:

```text
REP-004 RootSIFT+LBP
vs
REP-005 RootSIFT+LBP+HSV
```

If HSV improvement is:

```text
< +0.5 percentage point
```

do NOT carry HSV forward unless it improves:

```text
minimum(cat_accuracy, dog_accuracy)
```

by at least:

```text
+2.0 percentage points
```

Rationale:

- background contains unrelated colors;
- color must prove incremental value.

---

## Rule 3 — HOG / RootSIFT fusion

Do NOT automatically concatenate them.

Run an extra fusion candidate only if:

```text
best HOG-family candidate
AND
best RootSIFT-family candidate
```

are both within:

```text
2.0 percentage points
```

of the Phase-A best MLP validation accuracy.

If triggered, create:

```text
REP-006-FUSION
```

Components:

```text
best useful subset from:
HOG
RootSIFT
LBP if retained
HSV only if retained
```

Do not invent additional channels.

Run REP-006 with the same probes.

---

# 17. Phase-A candidate retention

Carry no more than TWO representations into Phase B.

Selection:

1. sort by fixed-MLP validation accuracy;
2. keep the best;
3. keep the runner-up if either:
   - accuracy difference <= 1.5 percentage points; or
   - runner-up uses at least 25% fewer flattened features.

If another candidate is within:

```text
0.5 percentage point
```

of the best but is dramatically smaller, prefer the smaller representation as the second confirmation candidate.

Do not carry more than two.

---

# 18. Phase B — multi-seed confirmation

Batch experiment ID:

```text
S1B-001
```

Representations:

```text
top two from Phase A only
+
REP-000-RAW64 control
```

Seeds:

```text
42
123
2026
```

For each seed:

```text
create/reuse stratified 90/10 split
fit per-channel normalization on train indices only
run LinearSVC
run fixed MLP
```

Total formal runs:

```text
3 representations x 3 seeds x 2 probes
```

Do not test additional representation hyperparameters here.

---

# 19. Phase-B winner selection

Primary metric:

```text
mean MLP validation accuracy across seeds
```

Also record:

```text
std
minimum-seed accuracy
mean cat accuracy
mean dog accuracy
mean balanced accuracy
mean macro F1
```

Selection rule:

If one candidate exceeds another by:

```text
>= 1.0 percentage point mean accuracy
```

choose the more accurate candidate.

If difference is:

```text
< 1.0 percentage point
```

choose the representation with lower:

```text
flatten dimension
```

unless the larger one has clearly better class balance:

```text
minimum(cat_accuracy,dog_accuracy) improvement >= 2.0 pp
```

---

# 20. Definition of Stage-I success

Use the three-seed `REP-000-RAW64` probe as the controlled reference.

Call handcrafted representation improvement "confirmed" only if:

```text
mean MLP validation accuracy
>= raw-control mean + 2.0 percentage points
```

Preferably also require:

```text
at least 2 of 3 seeds outperform raw control
```

Do not use the held-out 500-image test set to establish Stage-I success.

`DNN-001` remains separately reported as the official baseline.

---

# 21. Fallback if Stage I local maps fail

Trigger fallback only if the best confirmed handcrafted local map fails the Stage-I success criterion.

Do NOT trigger fallback merely because accuracy is lower than hoped.

Fallback order:

```text
Fallback 1:
Dense RootSIFT -> Spatial BoVW

Fallback 2:
RootSIFT -> Fisher Vector
```

These are vector-oriented representations.

If fallback is needed:

- implement them as additional DNN-oriented experiments;
- retain the canonical local RootSIFT feature map for CNN/RNN reuse;
- do not discard the spatial feature bank.

Do NOT implement BoVW/Fisher before the local-map experiments finish.

---

# 22. Optional compression experiment

Do NOT perform PCA during initial screening.

After the winning representation is selected, compression may be tested once if RootSIFT is present.

Example:

```text
RootSIFT128 -> PCA64
```

PCA rules:

```text
fit using training descriptors only
never fit on internal validation
never fit on data/val
whiten = False
random_state = split seed
```

Only retain PCA64 if:

```text
mean validation accuracy loss <= 0.5 pp
```

and feature dimension/storage is materially reduced.

Treat this as:

```text
S1C-001
```

not as part of initial feature discovery.

---

# 23. Runtime artifact structure

Large/generated artifacts:

```text
outputs/stage1/
├── cache/
│   ├── handcrafted128/
│   │   ├── features.npy
│   │   ├── labels.npy
│   │   ├── filenames.json
│   │   └── manifest.json
│   └── raw64/
│       └── ...
├── S1A-001/
│   ├── REP-001-HOG/
│   ├── REP-002-HOG-LBP/
│   └── ...
├── S1B-001/
│   ├── seed42/
│   ├── seed123/
│   └── seed2026/
└── diagnostics/
```

Keep all `outputs/` generated artifacts out of Git.

---

# 24. Per-run output contract

Every formal probe run must produce:

```text
config.json
normalization.json
metrics.json
history.csv              # MLP only
train_summary.json       # MLP only
predictions.csv
```

`predictions.csv` must contain internal-validation predictions only:

```text
filename
true_label
predicted_label
score_or_margin
correct
```

Purpose:

- later error overlap analysis;
- fusion justification;
- report examples;
- no need to rerun old experiments.

Do NOT save only aggregate accuracy.

---

# 25. Required metrics.json schema

At minimum:

```json
{
  "experiment_id": "",
  "batch_id": "",
  "representation_id": "",
  "probe": "",
  "seed": 42,
  "split_sha256": "",
  "representation_shape": [],
  "flatten_dim": 0,
  "parameter_count": null,
  "train_samples": 0,
  "validation_samples": 0,
  "best_epoch": null,
  "train_accuracy": null,
  "validation_accuracy": 0.0,
  "cat_accuracy": 0.0,
  "dog_accuracy": 0.0,
  "balanced_accuracy": 0.0,
  "macro_f1": 0.0,
  "confusion_matrix": [],
  "fit_seconds": 0.0,
  "feature_cache_manifest_sha256": "",
  "code_commit": ""
}
```

Populate actual values only.

---

# 26. Git-tracked report artifacts

Create:

```text
report/stage1/
├── stage1_results.csv
├── stage1_summary.md
├── representation_manifest.json
└── experiments/
    ├── S1A-001/
    ├── S1B-001/
    └── S1C-001/        # only if executed
```

Do NOT copy large feature arrays into `report/`.

For every formal batch, copy only compact artifacts needed for reproducibility:

```text
configs
metrics
summaries
split references/hashes
feature manifest
normalization stats
small CSV histories
```

---

# 27. stage1_results.csv

One row per formal probe run.

Required columns:

```text
batch_id
run_id
representation_id
probe
seed
channels
spatial_h
spatial_w
flatten_dim
parameter_count
validation_accuracy
cat_accuracy
dog_accuracy
balanced_accuracy
macro_f1
best_epoch
fit_seconds
feature_extraction_seconds
cache_bytes
split_sha256
git_commit
status
```

Do not edit this table manually.

Generate/rebuild it from per-run JSON artifacts.

---

# 28. stage1_summary.md

Generate automatically from `stage1_results.csv`.

Must contain:

```text
1. DNN-001 official baseline reference
2. preprocessing control result
3. Phase-A table
4. Phase-B mean ± std table
5. selected Stage-I representation
6. decision rationale based strictly on measured values
7. rejected feature groups and reason
8. representation shape
9. how DNN/CNN/RNN will consume the winner
10. unresolved limitations
```

No fabricated prose such as:

```text
"RootSIFT is better because ..."
```

unless actual experiments support it.

Use wording such as:

```text
"Under the three internal splits tested, REP-X achieved ..."
```

---

# 29. Prediction disagreement analysis

After Phase A, if HOG-family and RootSIFT-family candidates are both competitive, compare their validation predictions.

Calculate:

```text
A_correct_B_wrong
A_wrong_B_correct
both_correct
both_wrong
prediction_disagreement_rate
```

Use this analysis only to decide whether fusion is worth testing.

Do not introduce ensembling in Stage I.

---

# 30. Reuse by DNN, CNN and RNN

Implement one dataset adapter for cached feature maps.

Suggested location:

```text
representations/dataset.py
```

Conceptual API:

```python
CachedRepresentationDataset(
    features_path,
    labels_path,
    indices,
    channel_indices,
    channel_mean,
    channel_std,
)
```

Output:

```text
feature_map: Tensor[C,7,7]
label: int
```

Adapters:

## DNN

```python
x = feature_map.flatten()
```

## CNN

```python
x = feature_map
```

Input channels:

```text
C = selected representation channel count
```

Do not use a four-block image CNN unchanged on a 7x7 feature map.

Later CNN-on-representation experiments will require a shallow feature-map CNN.

## RNN

Preferred:

```python
[C,7,7]
-> permute/reshape
-> [7, 7*C]
```

Then:

```text
sequence_length = 7
input_size = 7*C
```

Alternative enhanced experiment:

```text
[49,C]
```

but keep row-sequence form as the primary representation-based RNN unless there is a documented reason otherwise.

---

# 31. Important fairness distinction for later experiments

Final report must distinguish:

```text
Raw-input architecture baselines:
DNN raw pixels
CNN raw pixels
RNN raw-image rows

Representation-enhanced variants:
DNN handcrafted feature map
CNN handcrafted feature map
RNN handcrafted feature map
```

Do not compare them as though preprocessing were identical.

This distinction is required for interpretability and assignment compliance.

---

# 32. Dependency changes

Allowed additions:

```text
scikit-image
scikit-learn
opencv-python-headless
```

Do not add:

```text
torchvision pretrained models
timm
transformers
CLIP
DINO
SAM
YOLO
other neural feature extractors
```

Update `requirements.txt` only with dependencies actually used.

Record exact runtime package versions in cache manifest.

---

# 33. Implementation order

Execute in exactly this order.

## Step 1

Inspect current Git status.

Do not destroy unrelated changes.

## Step 2

Implement canonical 128x128 letterbox preprocessing.

Run visual and shape smoke tests.

## Step 3

Implement HOG/LBP/HSV/RootSIFT atomic extractors.

Verify exact output shapes.

## Step 4

Implement master feature cache.

Build cache once for all 2000 `data/train` images.

Validate cache integrity.

## Step 5

Implement representation registry/channel slicing.

## Step 6

Implement split-specific channel normalization.

## Step 7

Implement cached representation dataset.

## Step 8

Implement LinearSVC probe.

## Step 9

Implement fixed MLP probe by reusing existing training utilities where reasonable.

Do not duplicate training logic unnecessarily.

## Step 10

Run tiny-overfit diagnostic.

## Step 11

Run `CTRL-LBOX64`.

## Step 12

Run Phase A `S1A-001`.

## Step 13

Generate Phase-A summary and apply decision rules mechanically.

## Step 14

Run conditional fusion only if Rule 3 triggers.

## Step 15

Select exactly two representations.

## Step 16

Create seeds 123 and 2026 splits.

## Step 17

Run Phase B `S1B-001`.

## Step 18

Generate final Stage-I summary and select one winning representation.

## Step 19

Do NOT evaluate `data/val`.

Stop Stage I.

Report the winner and recorded evidence.

---

# 34. CLI design

Prefer one explicit orchestration script rather than many ad-hoc commands.

Recommended:

```text
handcrafted_feature_experiments.py
```

Required commands conceptually:

```bash
python handcrafted_feature_experiments.py build-cache

python handcrafted_feature_experiments.py validate-cache

python handcrafted_feature_experiments.py run-control

python handcrafted_feature_experiments.py run-screen

python handcrafted_feature_experiments.py summarize-screen

python handcrafted_feature_experiments.py run-confirm

python handcrafted_feature_experiments.py summarize-confirm
```

Exact argparse subcommand implementation may differ slightly, but:

- commands must be documented;
- rerunning a completed cache should not silently overwrite incompatible artifacts;
- use config hashes to detect mismatch.

---

# 35. Cache invalidation

Compute a deterministic hash from:

```text
preprocessing config
feature extraction config
ordered source filenames
relevant code/config schema version
```

Before reusing cache:

```text
compare expected config hash with manifest
```

If mismatch:

```text
do not silently reuse cache
```

Create/rebuild the appropriate cache.

Do not rely only on directory names.

---

# 36. Failure handling

If a formal run fails:

record:

```text
status = failed
error summary
partial artifact paths
```

Do not fabricate metrics.

If a feature extractor produces NaN/Inf:

```text
stop before training
```

If a cache filename/label mismatch is detected:

```text
invalidate cache
stop
fix data alignment
rebuild
```

---

# 37. Do not optimize these during Stage I

Do NOT search:

```text
HOG orientation count
HOG cell size
LBP radius/P
HSV bin count
RootSIFT keypoint size
MLP depth
MLP hidden width
dropout
learning rate
weight decay
optimizer
scheduler
batch size
classification threshold
```

The fixed values in this specification are intentional.

Stage I asks:

```text
"Does the representation help?"
```

not:

```text
"What is the globally optimal pipeline?"
```

Architecture and optimizer optimization belongs to Stage II.

---

# 38. Stage-I completion criteria

Stage I is complete only when:

```text
[ ] DNN-001 preserved unchanged
[ ] data/val untouched
[ ] aspect-preserving preprocessing implemented
[ ] atomic feature extractors verified
[ ] master cache generated and validated
[ ] REP-000..REP-005 available
[ ] Phase-A screening completed
[ ] conditional fusion decision executed
[ ] exactly two candidates confirmed across 3 seeds
[ ] raw control confirmed across same 3 seeds
[ ] all metrics stored
[ ] validation predictions stored
[ ] stage1_results.csv generated
[ ] stage1_summary.md generated
[ ] one representation selected
[ ] DNN/CNN/RNN adapter documented
[ ] no Stage-II architecture optimization performed
```

At completion, stop and report:

```text
selected representation ID
shape [C,7,7]
flatten dimension
3-seed MLP mean ± std
3-seed class accuracies
raw-control improvement
LinearSVC diagnostic result
feature cache size
feature extraction time
rationale for retaining/dropping HOG/LBP/HSV/RootSIFT
whether PCA compression is worth testing next
```

Do not proceed to Stage II without a separate instruction.

---

# 39. Final priority order for autonomous decisions

When an unlisted minor decision arises, apply:

```text
1. prevent data leakage
2. preserve experiment reproducibility
3. preserve classifier-independent spatial representation
4. maximize measured validation performance
5. preserve comparability
6. minimize unnecessary experiment count
7. minimize feature dimension/storage
8. code elegance
```

If an issue materially changes:

```text
data protocol
representation definition
held-out test usage
assignment compliance
feature family
experiment selection criteria
```

do not improvise silently.

Stop that branch and explicitly report the issue.

For routine implementation details that do not change the above semantics, make the smallest reasonable decision and continue.
