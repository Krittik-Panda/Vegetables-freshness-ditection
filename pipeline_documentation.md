# Fruit Freshness Detection — Full Pipeline Documentation

> **Project:** `pipeline_v2`  
> **Goal:** Given an image of a fruit, automatically identify the fruit species and predict its freshness on a 0–100% scale.  
> **Run order:** `00_extract` → `01_train` → `02_evaluate` → `03_predict`

---

## High-Level Architecture

```
Raw Images (DataSet/)
        │
        ▼
┌─────────────────────────────────────┐
│  00_extract.ipynb                   │
│  Feature Extraction                 │
│  ├── EfficientNetB0 embedding (1280)│
│  └── Handcrafted features     (30) │
│       = X_fused (N × 1310)          │
└────────────────┬────────────────────┘
                 │  X_fused.npy, y.npy, fruit_type.npy
                 ▼
┌─────────────────────────────────────┐
│  01_train.ipynb                     │
│  Model Training                     │
│  ├── Global StandardScaler          │
│  ├── Fruit Branch (SVM-RBF)         │
│  │    └── RFE → SVC (fruit label)  │
│  ├── Freshness Branch               │
│  │    ├── CMI prefilter (top 300)   │
│  │    ├── RFE → feature space       │
│  │    ├── DASFS anchors per fruit   │
│  │    └── KNN anomaly model per fruit│
│  └── Saves 8 model artifacts        │
└────────────────┬────────────────────┘
                 │  *.pkl, *.npy models
                 ▼
┌─────────────────────────────────────┐
│  02_evaluate.ipynb                  │
│  Evaluation & Diagnostics           │
│  ├── Fruit classifier accuracy      │
│  ├── Per-fruit freshness accuracy   │
│  ├── DASFS projection plots         │
│  └── KNN distance distributions     │
└────────────────┬────────────────────┘
                 ▼
┌─────────────────────────────────────┐
│  03_predict.ipynb                   │
│  Inference                          │
│  ├── Single image prediction        │
│  └── Batch prediction (test_img/)   │
└─────────────────────────────────────┘
```

---

## Stage 0 — Dataset Fingerprinting (00_extract.ipynb)

Before any computation begins, the notebook computes an **MD5 hash** of the entire dataset (all image paths + file sizes) and compares it with the hash saved from the previous run.

```python
def compute_dataset_hash(folder: str) -> str:
    file_info = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if Path(f).suffix.lower() in EXTS:
                path = os.path.join(root, f)
                file_info.append(f"{path}:{os.path.getsize(path)}")
    raw = "|".join(sorted(file_info)).encode()
    return hashlib.md5(raw).hexdigest()
```

If the hash **matches** what is saved in `pipeline_v2/artifacts/dataset_hash.txt`, all cached `.npy` feature files are loaded directly and the extraction is skipped entirely — saving minutes of GPU compute. If the hash **differs** (new images added, existing images changed), re-extraction runs automatically.

---

## Stage 1 — Feature Extraction (00_extract.ipynb)

Each image produces a **1310-dimensional feature vector** formed by concatenating two complementary feature sets.

### 1A. Handcrafted Features (30 dimensions)

Extracted by `extract_handcrafted(img_bgr)`. The 30 features are organized into five groups:

| Group | Dimensions | Description |
|---|---|---|
| **RGB Statistics** | 6 | Per-channel mean and std of R, G, B |
| **HSV Statistics** | 6 | Circular mean of hue (via arctan2), saturation/value mean+std, circular spread |
| **LAB Statistics** | 6 | L*, a*, b* channel means and stds (perceptual colour space) |
| **Texture** | 5 | Laplacian variance (sharpness), GLCM contrast/energy/homogeneity, pixel intensity entropy |
| **Shape** | 6 | Contour area, perimeter, circularity, convexity ratio, aspect ratio, extent |
| **Dark Pixel Ratio** | 1 | Fraction of pixels below intensity 50 (browning/rotting indicator) |

All images are resized to **224×224** before feature extraction.

### 1B. EfficientNetB0 Deep Embedding (1280 dimensions)

A pretrained `EfficientNetB0` (ImageNet weights, no top layer, global average pooling) is loaded once and used for batch inference:

```python
backbone = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", pooling="avg"
)
```

Images are processed in batches of 64 (configurable). The output is a 1280-d embedding vector per image capturing high-level semantic features.

### 1C. Feature Fusion

The two feature sets are concatenated along axis-1:

```
X_fused = [EfficientNet embedding (1280) | Handcrafted (30)] → shape (N, 1310)
```

### Dataset Layout Expected

```
DataSet/
├── fresh/
│   ├── apple/     ← images
│   ├── banana/
│   └── ...
└── rotten/
    ├── apple/
    ├── banana/
    └── ...
```

The `label` is derived from the parent folder name: directories starting with `"fresh"` → label `0`; otherwise → label `1`. The fruit species name is parsed from the subfolder (with a name-cleaning step that strips the `fresh`/`rotten` prefix).

### Outputs Saved

| File | Shape | Description |
|---|---|---|
| `X_fused.npy` | (N, 1310) | Fused feature matrix |
| `y.npy` | (N,) | Binary freshness labels (0=fresh, 1=rotten) |
| `fruit_type.npy` | (N,) | Fruit species string per sample |
| `dataset_hash.txt` | — | MD5 fingerprint for cache invalidation |

---

## Stage 2 — Model Training (01_train.ipynb)

### 2A. Dataset Integrity Check

At startup, the notebook verifies that `X_fused.npy` and `dataset_hash.txt` both exist and are consistent. If either is missing, it raises a `FileNotFoundError` with a clear message. This prevents silent training on stale features.

### 2B. Fruit Name Cleaning

Because the dataset folder names encode both species and freshness (e.g. `"freshapple"`, `"rottenapple"`), a cleaning function strips the prefix so that `fruit_type` contains pure species names:

```python
def _clean_fruit_name(name: str) -> str:
    for prefix in ["fresh", "rotten"]:
        if name.startswith(prefix):
            tail = name[len(prefix):].lstrip("_")
            if tail:
                return tail
    return name
```

The cleaned array is re-saved to `fruit_type.npy`.

### 2C. Train/Test Split

An 80/20 stratified split is performed using a **combined stratification label** (`fruit_type + "_" + freshness_label`). This ensures every fruit × freshness cell is proportionally represented in both splits, which `sklearn`'s single-`stratify` argument requires.

```python
strat_label = [f"{ft}_{lbl}" for ft, lbl in zip(fruit_type, y)]
X_train, X_test, y_train, y_test, ft_train, ft_test = train_test_split(
    X, y, fruit_type, test_size=0.2, stratify=strat_label, random_state=42
)
```

### 2D. Global Scaling

A `StandardScaler` is fit **exclusively on the training split** and applied to both splits. This is the first transformation in the pipeline and all subsequent feature selection happens in scaled space.

```
scaler.pkl → global_scaler
```

### 2E. Fruit Branch — Feature Selection & Classification

**Feature Selection (RFE):** Recursive Feature Elimination with a `LinearSVC` estimator. Cross-validated over candidate counts `[80, 100, 120, 140]` with 5 stratified folds. The best count is selected by highest mean CV accuracy.

**Speed optimizations applied:**
- `LinearSVC(C=0.01, max_iter=1000, tol=1e-3)` — 13× faster than default params for ranking
- `joblib.Parallel(n_jobs=-1)` over all folds — additional ~5–8× speedup on multi-core machines

**Final classifier:** `SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", probability=True)`. `probability=True` is required for the confidence gate used at inference time.

**Outputs:**
- `rfe_fruit.pkl` — fitted RFE transformer for fruit feature selection
- `scaler_fruit.pkl` — secondary StandardScaler applied after RFE
- `svm_fruit.pkl` — trained RBF-SVM fruit classifier

### 2F. Freshness Branch — CMI Prefilter

Before RFE, a **Conditional Mutual Information (CMI)** score is computed for all 1310 features:

```
CMI_i = Σ_f  (N_f / N_total) · MI(X_i, freshness | fruit == f)
```

For each fruit species, `mutual_info_classif` is called once to compute MI between each feature and the freshness label (conditioned on that fruit). The per-fruit MIs are weighted by fruit prevalence and summed. Fruits with fewer than 10 training samples are skipped.

The **top 300 features** by CMI score are retained as the freshness prefilter output. This dramatically reduces the search space for the subsequent RFE step.

**Output:** `top300_cmi.npy`

### 2G. Freshness Branch — RFE Feature Selection

Same parallel CV-over-RFE procedure as the fruit branch, but applied on the 300-feature prefiltered space. Candidate counts: `[80, 100, 120, 140]`. The selected feature indices are mapped back to the original 1310-d space and saved.

**Outputs:**
- `rfe_fresh.pkl` — fitted RFE transformer for freshness feature selection
- `scaler_fresh.pkl` — secondary StandardScaler applied after RFE
- `fresh_final_idx.npy` — original-space indices of selected freshness features

### 2H. DASFS — Dual-Anchor Spectral Freshness Scoring

DASFS builds a **per-fruit freshness axis** in the freshness feature space. For each fruit species:

1. Compute the mean vector of all fresh samples (`μ_fresh`) and all rotten samples (`μ_rotten`)
2. Define the degradation axis: `axis = μ_rotten − μ_fresh`, normalized
3. Project all fresh and rotten training samples onto this axis
4. Record: `p_fresh` (median fresh projection), `p_rotten` (median rotten projection), `spread` (max of fresh/rotten std)

At inference, a new sample is projected onto the same axis and its position relative to `p_fresh` and `p_rotten` gives its freshness score:

```python
score = clip((p_rotten − proj) / (p_rotten − p_fresh), 0, 1)
```

A `sep` metric (`(p_rotten − p_fresh) / spread`) is printed as a separability diagnostic — higher values indicate better DASFS discriminability for that fruit.

**Output:** `dasfs.pkl` — dict keyed by fruit name, each containing `{axis, p_fresh, p_rotten, spread}`

### 2I. Per-Fruit KNN Anomaly Model

A `NearestNeighbors(k=5, metric="euclidean")` model is trained for each fruit using **only fresh samples** in the freshness feature space. A per-fruit threshold `τ` is computed as the median mean-neighbour-distance across fresh training samples.

At inference, the KNN support score for a new sample is:
```python
knn_sup = exp(−k_dist / τ)
```

High support (score near 1.0) means the sample is close to known-fresh samples; low support (near 0.0) means it is anomalous relative to the fruit's fresh distribution.

**Output:** `knn_dict.pkl` — dict with `knn_dict` (per-fruit KNN model) and `tau_dict` (per-fruit threshold)

### 2J. CV Results Persistence

After training, the CV results are serialized to `cv_results.json` so that `02_evaluate.ipynb` can plot the accuracy-vs-feature-count curves in any new session without needing to re-run training.

---

## Stage 3 — Evaluation (02_evaluate.ipynb)

Six diagnostic plots are generated and saved to `pipeline_v2/artifacts/`.

### Plot 1 — Accuracy vs. Feature Count
Shows the 5-fold CV accuracy for each candidate feature count in both the fruit and freshness RFE searches. The best count is highlighted in red. Loaded from `cv_results.json`.

### Plot 2 — Fruit Classifier Confusion Matrix
The RBF-SVM fruit classifier is run on the held-out test split. A full `classification_report` (precision, recall, F1 per species) is printed alongside a colour-coded confusion matrix heatmap.

### Plot 3 — Per-Fruit Freshness Accuracy
Uses **ground-truth** fruit labels (not predicted ones) to isolate freshness classification quality from fruit identification errors. For each fruit, three metrics are reported at DASFS threshold 0.50:

- **Fresh accuracy** — among truly-fresh samples, how many are predicted fresh
- **Rotten accuracy** — among truly-rotten samples, how many are predicted rotten  
- **Balanced accuracy** — average of the two

Displayed as a grouped bar chart with a 0.5 baseline.

### Plot 4 — DASFS Overall Freshness Accuracy
Aggregated over all fruits and test samples that have valid DASFS anchors. Reports overall accuracy, balanced accuracy, and a full classification report (`Fresh` vs `Rotten`).

### Plot 5 — KNN Distance Distribution per Fruit
For each fruit in the KNN dict, plots overlapping histograms of mean KNN distances for fresh (green) and rotten (red) test samples, with the τ threshold marked. A well-separated distribution indicates good anomaly detection.

### Plot 6 — DASFS Projection Distribution per Fruit
For each fruit in DASFS, plots overlapping histograms of the projection values onto the degradation axis for fresh/rotten test samples, with `p_fresh`, `p_rotten`, and midpoint annotated. Validates that the DASFS axis actually separates the two classes.

---

## Stage 4 — Inference (03_predict.ipynb)

### 4A. Model Loading

All 8 model artifacts and the EfficientNetB0 backbone are loaded into memory at session start. This is done once; subsequent calls to `predict()` reuse the loaded objects.

### 4B. Feature Extraction at Inference

The same 1310-d pipeline as in `00_extract.ipynb`:
```python
raw_features = concat([EfficientNetB0_embedding(img), handcrafted_30(img)])
```

### 4C. Feature Routing

```
raw (1310) → global_scaler → x_sc (1310)
                 ├── rfe_fruit.transform → scaler_fruit → x_fruit  (best_n_fruit)
                 └── x_sc[:, top300_idx] → rfe_fresh.transform → scaler_fresh → x_fresh (best_n_fresh)
```

### 4D. Fruit Identification with Confidence Gate (CHANGE 5)

```python
probs      = svm_fruit.predict_proba(x_fruit)[0]
top1_fruit = classes[argmax(probs)]
top1_prob  = max(probs)

if top1_prob >= FRUIT_CONF_THRESHOLD (0.70):
    fruit = top1_fruit                 # high-confidence path
else:
    fruit = _top2_fallback(...)        # CHANGE 6: low-confidence path
```

Without this gate, a misidentified fruit would propagate silently into the wrong DASFS anchors, producing meaningless freshness scores.

### 4E. Top-2 Fallback (CHANGE 6)

When fruit confidence is below the threshold, both the top-1 and top-2 candidate fruits are evaluated:

1. For each candidate: compute DASFS score, DASFS confidence, KNN support
2. Validate the projection falls within the full freshness range (CHANGE 4, see below)
3. Select the candidate with the highest combined confidence: `0.6 × d_conf + 0.4 × knn_sup`
4. If both candidates fail validation → fall back to top-1 anyway

### 4F. DASFS Validity Check (CHANGE 4)

Before trusting a DASFS score, the projection is checked to lie within the known freshness range for that fruit:

```python
# OLD (incorrect — rejects very-fresh or very-rotten samples at the tails):
abs(proj - midpoint) <= 2 * spread

# NEW (correct — accepts the full distribution range):
proj >= (p_fresh - 2 * spread)  AND  proj <= (p_rotten + 2 * spread)
```

The old check measured distance from the midpoint, which incorrectly rejected samples that were extremely fresh (projecting far on the fresh side of the axis).

### 4G. Final Freshness Score Computation

```python
# If DASFS confidence is low (< 0.3), blend with KNN:
if d_conf < 0.3:
    freshness = 0.6 * dasfs_score + 0.4 * knn_support

# Otherwise trust DASFS:
else:
    freshness = dasfs_score

confidence = 0.6 * d_conf + 0.4 * knn_sup
```

### 4H. Grading Thresholds

| Freshness Score | Label |
|---|---|
| > 75% | Very Fresh |
| 50–75% | Fresh |
| 25–50% | Slightly Degraded |
| < 25% | Rotten |

### 4I. Output Dictionary

```python
{
    "fruit":            "apple",
    "label":            "Fresh",
    "freshness_score":  78.4,   # 0–100%
    "dasfs_score":      80.1,
    "knn_support":      74.2,
    "confidence":       76.9,
    "low_conf_flag":    False   # True if fruit confidence gate triggered
}
```

### 4J. Batch Prediction

All images in `test_img/` are predicted sequentially. Outputs a summary table (filename, fruit, label, score, confidence, low-confidence flag), a per-fruit breakdown (count, mean/min/max score, label distribution), and a colour-coded bar chart saved to `pipeline_v2/artifacts/plot_batch_prediction.png`.

---

## Full Artifact Manifest

```
pipeline_v2/
├── models/
│   ├── scaler.pkl          Global StandardScaler (fit on train split)
│   ├── rfe_fruit.pkl       Fruit RFE transformer
│   ├── scaler_fruit.pkl    Secondary scaler for fruit branch
│   ├── svm_fruit.pkl       RBF-SVM fruit classifier (probability=True)
│   ├── rfe_fresh.pkl       Freshness RFE transformer
│   ├── scaler_fresh.pkl    Secondary scaler for freshness branch
│   ├── dasfs.pkl           Per-fruit DASFS anchors {axis, p_fresh, p_rotten, spread}
│   └── knn_dict.pkl        Per-fruit KNN models + tau thresholds
│
└── artifacts/
    ├── X_fused.npy              (N, 1310) — full fused features
    ├── y.npy                    (N,)      — freshness labels
    ├── fruit_type.npy           (N,)      — cleaned fruit species
    ├── dataset_hash.txt         MD5 fingerprint
    ├── X_fruit_train/test.npy   Fruit-branch features for train/test
    ├── X_fresh_train/test.npy   Freshness-branch features for train/test
    ├── y_train/test.npy         Labels for train/test splits
    ├── ft_train/test.npy        Fruit types for train/test splits
    ├── top300_cmi.npy           Indices of top-300 CMI-selected features
    ├── fresh_final_idx.npy      Final freshness feature indices (original space)
    ├── cv_results.json          CV accuracy results for plotting
    └── plot_*.png               All evaluation plots
```

---

## Key Design Decisions & Bug Fixes

| # | Change | Notebook | Impact |
|---|---|---|---|
| 1 | MD5 hash guard prevents training on stale features | 00, 01 | Correctness |
| 2 | Joint fruit×freshness stratification for train/test split | 01 | Data quality |
| 3 | Vectorised CMI per fruit (one `mutual_info_classif` call per fruit, not per feature) | 01 | Speed |
| 4 | DASFS validity check uses full range `[p_fresh−2σ, p_rotten+2σ]` not midpoint distance | 03 | Correctness |
| 5 | Fruit confidence gate (`≥0.70`) before propagating to DASFS | 03 | Correctness |
| 6 | Top-2 fallback with validity check for low-confidence fruit IDs | 03 | Robustness |
| 7 | `LinearSVC(C=0.01, max_iter=1000, tol=1e-3)` — 13× faster per fit | 01 | Speed |
| 8 | `joblib.Parallel` over CV folds — ~5–8× additional speedup | 01 | Speed |
| A | Strip `fresh`/`rotten` prefix from `fruit_type` (PATCH A) | 01 | Critical fix — DASFS was getting zero rotten samples without this |
| B | Persist `cv_results.json` so evaluation works across sessions | 01, 02 | Usability |
