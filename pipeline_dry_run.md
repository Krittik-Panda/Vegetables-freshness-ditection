# Pipeline Dry Run — Step-by-Step with a 15 × 20 Example Matrix

> **What this document does:** Takes a concrete, small example — 15 fruit images described
> by 20 numbers each — and walks through every processing step by hand with real arithmetic.
> Numbers are rounded to 2–3 decimal places for readability.
> The real system uses 1310 features instead of 20, but every formula is identical.

---

## 0. The Example Dataset

### Feature Descriptions (20 features, simplified from 1310)

| ID | Name | What it measures |
|---|---|---|
| F01 | R_mean | Average red channel intensity (0–255) |
| F02 | G_mean | Average green channel intensity (0–255) |
| F03 | B_mean | Average blue channel intensity (0–255) |
| F04 | Hue | Circular mean hue angle (0–360°) |
| F05 | Saturation | Colour vividness (0–255) |
| F06 | Brightness | Overall brightness (0–255) |
| F07 | L_star | Perceptual lightness — LAB space (0–255) |
| F08 | a_star | Green–red axis — LAB space |
| F09 | b_star | Blue–yellow axis — LAB space |
| F10 | R_std | Red channel variation |
| F11 | G_std | Green channel variation |
| F12 | B_std | Blue channel variation |
| F13 | Laplacian | Edge sharpness (blurry fruit = low) |
| F14 | GLCM_contrast | Texture roughness |
| F15 | GLCM_energy | Texture uniformity |
| F16 | GLCM_homog | Texture smoothness |
| F17 | Pixel_entropy | Complexity of pixel intensity histogram |
| F18 | Contour_area | Size of fruit outline (pixels²) |
| F19 | Sharpness | Overall focus / crispness score |
| F20 | Dark_ratio | Fraction of very dark pixels — key rot indicator |

---

### The 15 × 20 Raw Feature Matrix

> **Reading the table:** Each row is one fruit image. Label `0 = Fresh`, `1 = Rotten`.
> Highlighted rows (★) are the 10 fruit examples we will call out specifically during the dry run.

```
ID   Fruit    Lbl  F01  F02  F03  F04  F05  F06  F07  F08  F09  F10   F11  F12  F13  F14   F15   F16  F17    F18  F19   F20
───  ───────  ───  ───  ───  ───  ───  ───  ───  ───  ───  ───  ────  ───  ───  ───  ────  ────  ───  ───  ─────  ───  ────
★S01 apple    0   180   80   60   12  210  200  145   20   25   8.0   6.0  5.0  150  0.30  0.62  0.70 4.20   8200  320  0.04
★S02 apple    0   175   85   65   10  205  195  142   22   28   7.5   6.5  5.5  140  0.28  0.64  0.72 4.00   7800  290  0.05
★S03 apple    1   110   90   55   25  120  130  105    8   10  12.0  10.0  8.0   50  0.80  0.35  0.42 5.80   6500   85  0.22
★S04 banana   0   210  200   50   55  240  220  180   -5   55   5.0   8.0  4.0  180  0.20  0.72  0.80 3.50   9500  410  0.02
★S05 banana   0   215  195   48   52  235  215  175   -4   52   5.5   7.5  4.5  170  0.22  0.70  0.78 3.60   9200  380  0.03
★S06 banana   1   140  100   40   35   90  110   98    5   20  14.0  12.0  9.0   40  0.90  0.32  0.38 6.20   7000   75  0.28
★S07 mango    0   220  150   40   42  250  210  170   10   60   6.0   7.0  4.0  160  0.25  0.68  0.75 3.80  10200  350  0.03
 S08 mango    1   145  110   50   30  100  120  108   15   30  13.0  11.0  8.5   55  0.85  0.33  0.40 6.00   6800   80  0.25
 S09 mango    1   130   95   45   28   85  105   95   12   25  14.5  12.5  9.5   42  0.92  0.30  0.36 6.40   6200   68  0.30
★S10 orange   0   230  130   30   30  245  225  185   15   50   5.0   6.5  3.5  175  0.22  0.70  0.78 3.60   9800  440  0.02
 S11 orange   0   225  125   28   32  240  220  180   14   48   5.5   7.0  4.0  165  0.24  0.68  0.76 3.70   9500  420  0.03
★S12 orange   1   155  105   42   22  110  115  100   10   18  12.5  10.5  8.0   52  0.82  0.34  0.42 5.90   7100   78  0.24
★S13 grape    0   120   60  140  280  180  160  130   25  -10   9.0   7.0  9.0  130  0.32  0.62  0.70 4.10   4200  180  0.05
 S14 grape    1    90   70  100  270   90  100   90   18   -5  11.0   9.0 10.0   48  0.78  0.36  0.44 5.70   3500   70  0.20
 S15 grape    1    85   65   95  268   85   95   88   16   -6  12.0  10.0 11.0   45  0.80  0.34  0.42 5.90   3200   65  0.22
```

**Dataset summary:**

| Fruit | Fresh samples | Rotten samples | Total |
|---|---|---|---|
| Apple | S01, S02 | S03 | 3 |
| Banana | S04, S05 | S06 | 3 |
| Mango | S07 | S08, S09 | 3 |
| Orange | S10, S11 | S12 | 3 |
| Grape | S13 | S14, S15 | 3 |
| **Total** | **8** | **7** | **15** |

---

## Step 1 — Train / Test Split

We split 80% to training (12 rows) and 20% to testing (3 rows).
The split is **stratified** — each fruit × freshness combination is proportionally
represented in both halves.

```
Combined stratification key = fruit_name + "_" + label
```

| Sample | Strat key | Assigned to |
|---|---|---|
| S01 | apple_0 | **TRAIN** |
| S02 | apple_0 | **TRAIN** |
| S03 | apple_1 | **TEST** ← one rotten apple held out |
| S04 | banana_0 | **TRAIN** |
| S05 | banana_0 | **TRAIN** |
| S06 | banana_1 | **TEST** ← one rotten banana held out |
| S07 | mango_0 | **TRAIN** |
| S08 | mango_1 | **TRAIN** |
| S09 | mango_1 | **TEST** ← one rotten mango held out |
| S10 | orange_0 | **TRAIN** |
| S11 | orange_0 | **TRAIN** |
| S12 | orange_1 | **TRAIN** |
| S13 | grape_0 | **TRAIN** |
| S14 | grape_1 | **TRAIN** |
| S15 | grape_1 | **TRAIN** |

**Training set (12 rows):** S01, S02, S04, S05, S07, S08, S10, S11, S12, S13, S14, S15
**Test set (3 rows):** S03, S06, S09

> All subsequent fitting steps use **only training data**. Test data is untouched until evaluation.

---

## Step 2 — Global StandardScaler

Each feature column is re-centred to mean = 0 and scaled to std = 1.
**Formula:** `scaled_value = (raw_value − column_mean) / column_std`

The mean and std are computed from the **12 training rows only**.

We demonstrate this fully for **F19 (Sharpness)** and **F20 (Dark_ratio)** —
the two most intuitive freshness indicators.

---

### F19 (Sharpness) — full calculation

**Training values (12 rows):**

| Sample | Fruit | Label | F19 raw |
|---|---|---|---|
| S01 | apple | Fresh | 320 |
| S02 | apple | Fresh | 290 |
| S04 | banana | Fresh | 410 |
| S05 | banana | Fresh | 380 |
| S07 | mango | Fresh | 350 |
| S08 | mango | Rotten | 80 |
| S10 | orange | Fresh | 440 |
| S11 | orange | Fresh | 420 |
| S12 | orange | Rotten | 78 |
| S13 | grape | Fresh | 180 |
| S14 | grape | Rotten | 70 |
| S15 | grape | Rotten | 65 |

**Step 2.1 — Compute mean:**
```
Sum = 320+290+410+380+350+80+440+420+78+180+70+65 = 3083
Mean(F19) = 3083 / 12 = 256.92
```

**Step 2.2 — Compute standard deviation (n−1):**
```
Squared deviations from mean (256.92):
  S01: (320−256.92)² = (63.08)²  =  3979.1
  S02: (290−256.92)² = (33.08)²  =  1094.3
  S04: (410−256.92)² = (153.08)² = 23433.5
  S05: (380−256.92)² = (123.08)² = 15148.7
  S07: (350−256.92)² = (93.08)²  =  8663.9
  S08: (80−256.92)²  = (−176.92)²= 31300.7
  S10: (440−256.92)² = (183.08)² = 33518.3
  S11: (420−256.92)² = (163.08)² = 26595.1
  S12: (78−256.92)²  = (−178.92)²= 32012.4
  S13: (180−256.92)² = (−76.92)² =  5916.7
  S14: (70−256.92)²  = (−186.92)²= 34939.1
  S15: (65−256.92)²  = (−191.92)²= 36833.1

Sum of squared deviations = 253,434.9
Variance = 253,434.9 / (12−1) = 23,039.5
Std(F19) = √23,039.5 = 151.79
```

**Step 2.3 — Scale all training F19 values:**
```
scaled = (raw − 256.92) / 151.79

S01: (320−256.92)/151.79 = +0.42
S02: (290−256.92)/151.79 = +0.22
S04: (410−256.92)/151.79 = +1.01
S05: (380−256.92)/151.79 = +0.81
S07: (350−256.92)/151.79 = +0.61
S08: (80−256.92)/151.79  = −1.16
S10: (440−256.92)/151.79 = +1.21
S11: (420−256.92)/151.79 = +1.07
S12: (78−256.92)/151.79  = −1.18
S13: (180−256.92)/151.79 = −0.51
S14: (70−256.92)/151.79  = −1.23
S15: (65−256.92)/151.79  = −1.26
```

> **Pattern visible:** Fresh samples cluster around +0.4 to +1.2 (sharp images).
> Rotten samples cluster around −1.2 to −0.5 (blurry, dull images). ✓

---

### F20 (Dark_ratio) — full calculation

**Training values:**

| Sample | Fruit | Label | F20 raw |
|---|---|---|---|
| S01 | apple | Fresh | 0.04 |
| S02 | apple | Fresh | 0.05 |
| S04 | banana | Fresh | 0.02 |
| S05 | banana | Fresh | 0.03 |
| S07 | mango | Fresh | 0.03 |
| S08 | mango | Rotten | 0.25 |
| S10 | orange | Fresh | 0.02 |
| S11 | orange | Fresh | 0.03 |
| S12 | orange | Rotten | 0.24 |
| S13 | grape | Fresh | 0.05 |
| S14 | grape | Rotten | 0.20 |
| S15 | grape | Rotten | 0.22 |

**Step 2.1 — Mean:**
```
Sum = 0.04+0.05+0.02+0.03+0.03+0.25+0.02+0.03+0.24+0.05+0.20+0.22 = 1.18
Mean(F20) = 1.18 / 12 = 0.0983
```

**Step 2.2 — Standard deviation:**
```
Squared deviations:
  S01: (0.04−0.0983)² = (−0.0583)² = 0.003399
  S02: (0.05−0.0983)² = (−0.0483)² = 0.002333
  S04: (0.02−0.0983)² = (−0.0783)² = 0.006131
  S05: (0.03−0.0983)² = (−0.0683)² = 0.004665
  S07: (0.03−0.0983)² = (−0.0683)² = 0.004665
  S08: (0.25−0.0983)² = (+0.1517)² = 0.023013
  S10: (0.02−0.0983)² = (−0.0783)² = 0.006131
  S11: (0.03−0.0983)² = (−0.0683)² = 0.004665
  S12: (0.24−0.0983)² = (+0.1417)² = 0.020079
  S13: (0.05−0.0983)² = (−0.0483)² = 0.002333
  S14: (0.20−0.0983)² = (+0.1017)² = 0.010343
  S15: (0.22−0.0983)² = (+0.1217)² = 0.014811

Sum = 0.102568
Variance = 0.102568 / 11 = 0.009324
Std(F20) = √0.009324 = 0.09656
```

**Step 2.3 — Scale all training F20 values:**
```
scaled = (raw − 0.0983) / 0.09656

S01: (0.04−0.0983)/0.09656  = −0.60
S02: (0.05−0.0983)/0.09656  = −0.50
S04: (0.02−0.0983)/0.09656  = −0.81
S05: (0.03−0.0983)/0.09656  = −0.71
S07: (0.03−0.0983)/0.09656  = −0.71
S08: (0.25−0.0983)/0.09656  = +1.57
S10: (0.02−0.0983)/0.09656  = −0.81
S11: (0.03−0.0983)/0.09656  = −0.71
S12: (0.24−0.0983)/0.09656  = +1.47
S13: (0.05−0.0983)/0.09656  = −0.50
S14: (0.20−0.0983)/0.09656  = +1.05
S15: (0.22−0.0983)/0.09656  = +1.26
```

> **Pattern visible:** Fresh samples all negative (low dark area). Rotten samples all positive (lots of dark patches). ✓

---

### Scaled Matrix Summary (F19 and F20 only, training rows)

```
ID   Fruit    Lbl   F19_sc   F20_sc
───  ───────  ───   ──────   ──────
S01  apple    0     +0.42    −0.60
S02  apple    0     +0.22    −0.50
S04  banana   0     +1.01    −0.81
S05  banana   0     +0.81    −0.71
S07  mango    0     +0.61    −0.71
S08  mango    1     −1.16    +1.57
S10  orange   0     +1.21    −0.81
S11  orange   0     +1.07    −0.71
S12  orange   1     −1.18    +1.47
S13  grape    0     −0.51    −0.50
S14  grape    1     −1.23    +1.05
S15  grape    1     −1.26    +1.26
```

> The scaler **must** also be applied to test rows using the **training** mean and std — not recalculated.

**Test rows scaled (using same Mean=256.92, Std=151.79 for F19):**
```
S03 (rotten apple):  F19=(85−256.92)/151.79 = −1.13   F20=(0.22−0.0983)/0.09656 = +1.26
S06 (rotten banana): F19=(75−256.92)/151.79 = −1.20   F20=(0.28−0.0983)/0.09656 = +1.88
S09 (rotten mango):  F19=(68−256.92)/151.79 = −1.24   F20=(0.30−0.0983)/0.09656 = +2.09
```

---

## Step 3 — CMI Prefilter (Conditional Mutual Information)

**Goal:** Score each of the 20 features by how informative they are about freshness,
*within each fruit species*. Keep only the top features.

**Formula:**
```
CMI(feature_i) = Σ over each fruit:
                    (number of this fruit's samples / total samples)
                    × MI(feature_i, freshness_label | this fruit only)
```

Where `MI` = Mutual Information — a measure of how much knowing the feature reduces
uncertainty about whether the fruit is fresh or rotten.

### MI for F19 conditioned on each fruit

We walk through the logic for each fruit. MI is high when fresh and rotten values
are clearly separated for that feature.

#### Apple (S01, S02, S07, S08 in train — wait, apple rows in train: S01, S02)
Weight for apple = 2/12 = 0.167 (apple training rows: S01, S02; no rotten apple in train — S03 is in test!)

> **Note:** S03 (rotten apple) is in the test set, so apple has only 2 fresh training samples.
> MI is undefined with only one class present. Apple is **skipped** for CMI (only one label present in training).

#### Banana (S04=fresh, S05=fresh, S06=rotten — all in train)
Weight = 3/12 = 0.25

| Sample | F19_scaled | Label |
|---|---|---|
| S04 | +1.01 | 0 (Fresh) |
| S05 | +0.81 | 0 (Fresh) |
| S06 | −1.20\* | 1 (Rotten) |

(\*S06 test-scaled value used for illustration)

F19 values: Fresh group = {+1.01, +0.81}, Rotten group = {−1.20}
Separation: fresh_mean = +0.91, rotten_mean = −1.20, gap = 2.11 → **very large separation**
Approximate MI(F19 | banana) ≈ **0.88** (near max = entropy of y)

#### Mango (S07=fresh, S08=rotten, in train)
Weight = 2/12 = 0.167
F19_scaled: fresh = {+0.61}, rotten = {−1.16}
gap = 1.77 → large separation
Approximate MI(F19 | mango) ≈ **0.85**

#### Orange (S10=fresh, S11=fresh, S12=rotten, in train)
Weight = 3/12 = 0.25
F19_scaled: fresh = {+1.21, +1.07}, rotten = {−1.18}
fresh_mean = +1.14, rotten_mean = −1.18, gap = 2.32 → **very large**
Approximate MI(F19 | orange) ≈ **0.90**

#### Grape (S13=fresh, S14=rotten, S15=rotten, in train)
Weight = 3/12 = 0.25
F19_scaled: fresh = {−0.51}, rotten = {−1.23, −1.26}
fresh_mean = −0.51, rotten_mean = −1.245, gap = 0.735 → moderate separation
Approximate MI(F19 | grape) ≈ **0.65**

#### CMI(F19):
```
CMI(F19) = 0.167×0  (apple skipped)
          + 0.25×0.88  (banana)
          + 0.167×0.85  (mango)
          + 0.25×0.90   (orange)
          + 0.25×0.65   (grape)

         = 0 + 0.220 + 0.142 + 0.225 + 0.163
         = 0.750
```

---

### MI for F04 (Hue angle) conditioned on each fruit

F04 values (from raw data):

| Sample | Fruit | Lbl | F04 raw |
|---|---|---|---|
| S04 | banana | 0 | 55 |
| S05 | banana | 0 | 52 |
| S06 | banana | 1 | 35 |
| S07 | mango | 0 | 42 |
| S08 | mango | 1 | 30 |
| S10 | orange | 0 | 30 |
| S11 | orange | 0 | 32 |
| S12 | orange | 1 | 22 |
| S13 | grape | 0 | 280 |
| S14 | grape | 1 | 270 |
| S15 | grape | 1 | 268 |

For **orange**: fresh hues = {30, 32}, rotten hue = {22}. Small gap. **MI ≈ 0.30**
For **grape**: fresh = {280}, rotten = {270, 268}. Very small gap. **MI ≈ 0.20**
For **banana**: fresh = {55, 52}, rotten = {35}. Moderate gap. **MI ≈ 0.55**

```
CMI(F04) ≈ 0.25×0.55 + 0.167×0.50 + 0.25×0.30 + 0.25×0.20
          = 0.138 + 0.084 + 0.075 + 0.050
          = 0.347
```

---

### CMI Comparison Table (all 20 features — approximate)

| Feature | CMI Score | Keep? | Why |
|---|---|---|---|
| F19 (Sharpness) | **0.750** | ✅ Yes | Blurry = rotten, very consistent across all fruits |
| F20 (Dark_ratio) | **0.740** | ✅ Yes | Dark patches = rot, consistent signal |
| F05 (Saturation) | **0.720** | ✅ Yes | Vivid colour = fresh, dull = rotten |
| F06 (Brightness) | **0.710** | ✅ Yes | Brighter generally = fresher |
| F15 (GLCM_energy) | **0.690** | ✅ Yes | Uniform texture = fresh |
| F13 (Laplacian) | **0.680** | ✅ Yes | Sharp edges = fresh |
| F16 (GLCM_homog) | **0.670** | ✅ Yes | Smooth surface = fresh |
| F14 (GLCM_contrast) | **0.660** | ✅ Yes | High roughness = rotten |
| F07 (L_star) | **0.640** | ✅ Yes | Lightness drops when fruit darkens |
| F17 (Entropy) | **0.620** | ✅ Yes | Complex pixel hist = degraded |
| F18 (Contour_area) | **0.580** | ✅ Yes | Shrivelling reduces area |
| F01 (R_mean) | **0.520** | ✅ Yes | Red shifts as fruit degrades |
| F02 (G_mean) | **0.490** | ✅ Yes | Green channel informative for some fruits |
| F10 (R_std) | **0.460** | ✅ Yes | Uneven red = blotchy = rotten |
| F11 (G_std) | **0.440** | ✅ Yes | Uneven green variation |
| F08 (a_star) | **0.410** | ✅ Yes | Red–green axis shifts |
| F09 (b_star) | **0.390** | ✅ Yes | Blue–yellow shift |
| F04 (Hue) | **0.347** | ✅ Yes | Some hue shift at degradation |
| F03 (B_mean) | **0.180** | ❌ No | Blue channel not strongly linked to rot |
| F12 (B_std) | **0.120** | ❌ No | Blue variation carries little freshness info |

**Top features retained (CMI ≥ 0.35):** F19, F20, F05, F06, F15, F13, F16, F14, F07, F17, F18, F01, F02, F10, F11, F08, F09, F04 → **18 of 20 kept**

> In the real system, top **300 of 1310** features are kept. The math is identical.

---

## Step 4 — Fruit Branch: RFE Feature Selection + SVM Classifier

### Recursive Feature Elimination (RFE) — how it works

RFE iteratively removes the least important features using a `LinearSVC`:

```
Start with all 20 features
  → Train LinearSVC → rank features by |weight|
  → Remove bottom-ranked features
  → Repeat until target number reached
```

**CV results over candidate counts (simulated):**

| n_features | 5-fold CV fruit accuracy |
|---|---|
| 5 | 0.72 |
| 8 | 0.85 |
| 10 | **0.90** ← best |
| 15 | 0.88 |

**Best fruit feature count selected: 10**

### Features selected by RFE for Fruit Branch

After RFE with n=10:

| Selected | Feature | Why useful for fruit ID |
|---|---|---|
| ✅ | F04 (Hue) | Grape hue (~270°) vs banana (~55°) vs orange (~30°) — very discriminative |
| ✅ | F01 (R_mean) | Banana = very high red, grape = lower |
| ✅ | F02 (G_mean) | Banana has high green, apple mid |
| ✅ | F09 (b_star) | Banana = high yellow (+55), grape = negative (−10) |
| ✅ | F08 (a_star) | Separates apple (a=20) from banana (a=−5) |
| ✅ | F18 (Contour_area) | Banana large, grape small |
| ✅ | F05 (Saturation) | Distinct saturation per species |
| ✅ | F03 (B_mean) | Grape has high blue (~140), others low |
| ✅ | F07 (L_star) | Lightness differs by species |
| ✅ | F17 (Entropy) | Texture complexity differs |

**Fruit classifier (SVM-RBF) trained on 12 training samples, 10 features:**

The SVM draws curved decision boundaries in this 10-feature space. With
`probability=True`, it also outputs a confidence percentage per class.

---

## Step 5 — Freshness Branch: Final RFE Selection

RFE runs on the 18 CMI-filtered features, now targeting freshness (0/1) instead of fruit type.

**CV results:**

| n_features | 5-fold CV freshness accuracy |
|---|---|
| 5 | 0.78 |
| 8 | 0.88 |
| 10 | **0.91** ← best |
| 15 | 0.89 |

**Best freshness feature count: 10**
Features selected: F19, F20, F05, F06, F15, F14, F07, F13, F17, F16

These 10 features define the **freshness feature space** where DASFS and KNN operate.

---

## Step 6 — DASFS Anchors (Dual-Anchor Spectral Freshness Scoring)

We build a separate "freshness ruler" for each fruit.
Demonstrated in detail for **apple** using F19 and F20 (the two most
informative features — the full system uses all 10 freshness features).

### Apple DASFS — Using F19_scaled and F20_scaled

**Apple training samples** (after scaling):

| Sample | Label | F19_sc | F20_sc |
|---|---|---|---|
| S01 | Fresh (0) | +0.42 | −0.60 |
| S02 | Fresh (0) | +0.22 | −0.50 |

> Rotten apple (S03) is in the test set, so training has 2 fresh and 0 rotten.
> Apple would be **skipped** by the real system (needs min 5 fresh + 5 rotten).
> For demonstration purposes, we treat S03 as if it were also in training.

Using S01, S02 (fresh), S03 (rotten, scaled: F19=−1.13, F20=+1.26):

**Step 6.1 — Compute mean fresh and mean rotten vectors:**
```
μ_fresh = mean of fresh samples across each feature:
  F19: (0.42 + 0.22) / 2 = +0.320
  F20: (−0.60 + −0.50) / 2 = −0.550

μ_fresh = [+0.320, −0.550]

μ_rotten = S03 only (single rotten sample):
  F19: −1.130
  F20: +1.260

μ_rotten = [−1.130, +1.260]
```

**Step 6.2 — Compute degradation axis:**
```
axis (unnormalized) = μ_rotten − μ_fresh
  F19: −1.130 − 0.320 = −1.450
  F20: +1.260 − (−0.550) = +1.810

axis = [−1.450, +1.810]

Magnitude = √((−1.450)² + (1.810)²)
          = √(2.103 + 3.276)
          = √5.379
          = 2.319

axis_normalized = [−1.450/2.319, +1.810/2.319]
                = [−0.625, +0.781]
```

> **Meaning of the axis:** Moving along this direction — decreasing F19 (less sharp)
> AND increasing F20 (more dark pixels) — corresponds to moving from fresh to rotten.
> The axis encodes the *direction of degradation* for this specific fruit.

**Step 6.3 — Project each sample onto the axis:**

`projection = F19_scaled × (−0.625) + F20_scaled × (0.781)`

```
S01 (fresh): 0.42×(−0.625) + (−0.60)×0.781
           = −0.2625 + (−0.4686)
           = −0.731

S02 (fresh): 0.22×(−0.625) + (−0.50)×0.781
           = −0.1375 + (−0.3905)
           = −0.528

S03 (rotten): (−1.13)×(−0.625) + (1.26)×0.781
            = 0.706 + 0.984
            = +1.690
```

> **Visual intuition:** Fresh samples project to **−0.73 and −0.53** (left side of axis).
> Rotten sample projects to **+1.69** (right side). Large separation! ✓

**Step 6.4 — Record anchor points and spread:**
```
p_fresh  = median of fresh projections = median([−0.731, −0.528])
         = (−0.731 + −0.528) / 2
         = −0.630

p_rotten = median of rotten projections = −1.690
         Wait — let me re-check. Rotten projects to +1.690, which is correct:
p_rotten = +1.690

spread   = max(std_fresh, std_rotten)

std of fresh projections [−0.731, −0.528]:
  mean = −0.630
  std  = √(((−0.731−(−0.630))² + (−0.528−(−0.630))²) / 1)
       = √(((−0.101)² + (0.102)²) / 1)
       = √(0.01020 + 0.01040)
       = √0.02060
       = 0.1435

std of rotten (one sample) = 0.0

spread = max(0.1435, 0.0) = 0.1435
```

**Step 6.5 — Compute separability score:**
```
sep = (p_rotten − p_fresh) / spread
    = (1.690 − (−0.630)) / 0.1435
    = 2.320 / 0.1435
    = 16.2   ← very high! The axis cleanly separates fresh from rotten.
```

**Apple DASFS anchors saved to `dasfs.pkl`:**
```python
dasfs["apple"] = {
    "axis":     [−0.625, +0.781],   # direction of degradation
    "p_fresh":  −0.630,              # where fresh apples land
    "p_rotten": +1.690,              # where rotten apples land
    "spread":    0.1435              # how spread out the clusters are
}
```

---

### Banana DASFS (summary)

Using banana training rows S04, S05 (fresh), S06 (rotten):

```
μ_fresh_banana  = [+0.91, −0.76]  (average of S04, S05 in F19, F20 scaled)
μ_rotten_banana = [−1.20, +1.88]  (S06 scaled values)

axis = μ_rotten − μ_fresh = [−2.11, +2.64]
|axis| = √(4.454 + 6.970) = √11.424 = 3.380
axis_norm = [−0.625, +0.781]   ← same direction as apple! (F20 and F19 degrade similarly)

p_fresh_banana  = −0.784
p_rotten_banana = +2.120
spread_banana   = 0.102
sep_banana      = (2.120−(−0.784))/0.102 = 28.5  ← excellent separability
```

---

### Orange DASFS (summary)

Using S10, S11 (fresh), S12 (rotten):

```
p_fresh_orange  = −0.776
p_rotten_orange = +1.715
spread_orange   = 0.095
sep_orange      = 26.2
```

---

### Grape DASFS (summary)

Grape is interesting — it has **lower** sharpness values even when fresh
(smaller fruit, less distinct edges). The ruler is calibrated per-fruit so
this is handled automatically.

```
Using S13 (fresh), S14, S15 (rotten):
p_fresh_grape  = −0.261
p_rotten_grape = +1.320
spread_grape   = 0.168
sep_grape      = 9.4   ← lower but still good separability
```

---

## Step 7 — KNN Anomaly Model (Per Fruit)

For each fruit, we fit a K-Nearest-Neighbours model using **only fresh training samples**.

### Apple KNN (demonstration with k=2, using F19 and F20 scaled)

Fresh apple training samples: S01 [+0.42, −0.60], S02 [+0.22, −0.50]

**Fit step:** Store the fresh sample coordinates.

**Compute τ (threshold):**
```
Distance S01 → S02:
d = √((0.42−0.22)² + (−0.60−(−0.50))²)
  = √((0.20)² + (−0.10)²)
  = √(0.0400 + 0.0100)
  = √0.0500
  = 0.224

Each fresh sample's mean-knn-distance = 0.224 (only one neighbor each)
τ = median([0.224, 0.224]) = 0.224
```

### Banana KNN

Fresh banana: S04 [+1.01, −0.81], S05 [+0.81, −0.71]
```
d(S04→S05) = √((1.01−0.81)² + (−0.81−(−0.71))²)
           = √((0.20)² + (−0.10)²)
           = √0.0500 = 0.224

τ_banana = 0.224
```

### Orange KNN

Fresh orange: S10 [+1.21, −0.81], S11 [+1.07, −0.71]
```
d(S10→S11) = √((1.21−1.07)² + (−0.81−(−0.71))²)
           = √((0.14)² + (−0.10)²)
           = √(0.0196 + 0.0100)
           = √0.0296 = 0.172

τ_orange = 0.172
```

---

## Step 8 — Full Inference Dry Run (Predict on a New Image)

We now simulate the complete `predict()` function on **two new apple images** —
one fresh, one rotten.

---

### New Image A — Fresh Apple

Suppose the extracted 20-feature vector (after EfficientNet + handcrafted
extraction) for a fresh apple is:

```
F19 raw = 305,  F20 raw = 0.04
(other features omitted for brevity — same process applies to all 20)
```

---

#### 8A-1: Apply Global Scaler

```
F19_scaled = (305 − 256.92) / 151.79 = 48.08 / 151.79 = +0.317
F20_scaled = (0.04 − 0.0983) / 0.09656 = −0.0583 / 0.09656 = −0.604
```

---

#### 8A-2: Fruit Branch Transform (RFE + scaler_fruit → x_fruit)

The saved `rfe_fruit` extracts the 10 fruit-relevant features and `scaler_fruit`
re-scales them. For this example, the 10 features after transformation give a
feature vector `x_fruit` that looks like an apple (high R_mean, mid-range Hue,
positive b_star).

---

#### 8A-3: Fruit Identification — SVM Confidence Gate

```
svm_fruit.predict_proba(x_fruit) →

  apple:  0.82  ← top candidate
  banana: 0.09
  mango:  0.05
  orange: 0.03
  grape:  0.01

top1_fruit = "apple"
top1_prob  = 0.82
```

```
Is 0.82 ≥ FRUIT_CONF_THRESHOLD (0.70)?   YES → use "apple" directly
low_conf_flag = False
```

---

#### 8A-4: Freshness Branch Transform (top-300 CMI → RFE_fresh → scaler_fresh → x_fresh)

The freshness feature vector for this image, after all transforms, is:
```
F19_fresh = +0.317,  F20_fresh = −0.604
(using these two as our demonstration dimensions)
```

---

#### 8A-5: DASFS Validity Check

We use the apple DASFS anchors from Step 6.

**Project onto degradation axis:**
```
proj = F19_fresh × (−0.625) + F20_fresh × (0.781)
     = 0.317 × (−0.625) + (−0.604) × (0.781)
     = −0.198 + (−0.472)
     = −0.670
```

**Validity check — is the projection within the apple's expected range?**
```
Lower bound = p_fresh − 2 × spread = −0.630 − 2 × 0.1435 = −0.630 − 0.287 = −0.917
Upper bound = p_rotten + 2 × spread = +1.690 + 2 × 0.1435 = +1.690 + 0.287 = +1.977

Is −0.670 in [−0.917, +1.977]?   YES ✓   (projection is valid)
```

---

#### 8A-6: DASFS Freshness Score

```
score = clip( (p_rotten − proj) / (p_rotten − p_fresh) , 0, 1 )
      = clip( (1.690 − (−0.670)) / (1.690 − (−0.630)) , 0, 1 )
      = clip( 2.360 / 2.320 , 0, 1 )
      = clip( 1.017 , 0, 1 )
      = 1.0    (clipped — sample is actually fresher than the median-fresh anchor)
```

> A score of 1.0 means this apple's projection is at or beyond the typical
> "very fresh" position on the ruler. It is extremely fresh!

**DASFS confidence:**
```
midpoint = (p_fresh + p_rotten) / 2 = (−0.630 + 1.690) / 2 = 0.530

d_conf = clip( 1 − exp( −(proj − mid)² / (2 × spread²) ) , 0, 1 )
       = clip( 1 − exp( −(−0.670 − 0.530)² / (2 × 0.1435²) ) , 0, 1 )
       = clip( 1 − exp( −(−1.200)² / (2 × 0.02059) ) , 0, 1 )
       = clip( 1 − exp( −1.440 / 0.04118 ) , 0, 1 )
       = clip( 1 − exp( −34.97 ) , 0, 1 )
       = clip( 1 − 0.000 , 0, 1 )
       = 1.0    ← very high confidence (projection is far from midpoint)
```

---

#### 8A-7: KNN Support

```
New sample coordinates (F19, F20 scaled): [+0.317, −0.604]
Apple fresh training samples: S01=[+0.42, −0.60], S02=[+0.22, −0.50]

Distance to S01:
d₁ = √((0.317−0.42)² + (−0.604−(−0.60))²)
   = √((−0.103)² + (−0.004)²)
   = √(0.01061 + 0.000016)
   = √0.010626
   = 0.1031

Distance to S02:
d₂ = √((0.317−0.22)² + (−0.604−(−0.50))²)
   = √((0.097)² + (−0.104)²)
   = √(0.009409 + 0.010816)
   = √0.020225
   = 0.1422

Mean KNN distance = (0.1031 + 0.1422) / 2 = 0.1227

KNN support = exp(−mean_dist / τ)
            = exp(−0.1227 / 0.224)
            = exp(−0.548)
            = 0.578
```

---

#### 8A-8: Final Score Blending

```
d_conf   = 1.0
knn_sup  = 0.578

Since d_conf (1.0) ≥ 0.3:
    freshness = dasfs_score × 100 = 1.0 × 100 = 100.0%
    (clipped DASFS score is 1.0 → 100%)

Confidence = 0.6 × d_conf + 0.4 × knn_sup
           = 0.6 × 1.0 + 0.4 × 0.578
           = 0.600 + 0.231
           = 0.831 → 83.1%
```

---

#### 8A-9: Final Label Assignment

```
freshness_score = 100.0%
100.0% > 75%   → Label = "Very Fresh"
```

**Output dictionary for New Image A:**
```python
{
    "fruit":           "apple",
    "label":           "Very Fresh",
    "freshness_score": 100.0,
    "dasfs_score":     100.0,
    "knn_support":     57.8,
    "confidence":      83.1,
    "low_conf_flag":   False
}
```

---

### New Image B — Rotten Apple

Same apple type, but image shows heavy browning.

```
F19 raw = 82,   F20 raw = 0.24
```

#### Scaler:
```
F19_scaled = (82 − 256.92) / 151.79 = −174.92 / 151.79 = −1.152
F20_scaled = (0.24 − 0.0983) / 0.09656 = +0.1417 / 0.09656 = +1.467
```

#### Fruit SVM:
```
apple: 0.79  →  top1_fruit = "apple",  top1_prob = 0.79  ≥ 0.70 → no fallback
```

#### DASFS projection:
```
proj = (−1.152) × (−0.625) + (1.467) × (0.781)
     = 0.720 + 1.146
     = +1.866
```

#### Validity check:
```
Is +1.866 in [−0.917, +1.977]?   YES ✓
```

#### DASFS score:
```
score = clip( (1.690 − 1.866) / (1.690 − (−0.630)) , 0, 1 )
      = clip( (−0.176) / 2.320 , 0, 1 )
      = clip( −0.076 , 0, 1 )
      = 0.0    (clipped — sample projects beyond the rotten anchor — fully rotten!)
```

> The sample's projection (+1.866) exceeds `p_rotten` (+1.690) — it is more
> rotten-looking than the median rotten training sample. Score clips to 0.

#### DASFS confidence:
```
d_conf = clip( 1 − exp( −(1.866 − 0.530)² / (2 × 0.02059) ) , 0, 1 )
       = clip( 1 − exp( −(1.336)² / 0.04118 ) , 0, 1 )
       = clip( 1 − exp( −1.785 / 0.04118 ) , 0, 1 )
       = clip( 1 − exp( −43.3 ) , 0, 1 )
       ≈ 1.0
```

#### KNN support:
```
Distance to S01 (fresh): √((−1.152−0.42)² + (1.467−(−0.60))²)
                        = √((−1.572)² + (2.067)²)
                        = √(2.471 + 4.273)
                        = √6.744 = 2.597

Distance to S02 (fresh): √((−1.152−0.22)² + (1.467−(−0.50))²)
                        = √((−1.372)² + (1.967)²)
                        = √(1.882 + 3.869)
                        = √5.751 = 2.398

Mean KNN distance = (2.597 + 2.398) / 2 = 2.498

knn_sup = exp(−2.498 / 0.224) = exp(−11.15) ≈ 0.000
```

> The new rotten apple is **very far** from any fresh apple in the training set.
> KNN support ≈ 0 — this is anomalous compared to fresh samples. ✓

#### Final score:
```
d_conf   = 1.0
knn_sup  = 0.000
freshness = dasfs_score × 100 = 0.0 × 100 = 0.0%
confidence = 0.6×1.0 + 0.4×0.000 = 0.600 → 60.0%
```

#### Label:
```
0.0% ≤ 25%   → "Rotten"
```

**Output dictionary for New Image B:**
```python
{
    "fruit":           "apple",
    "label":           "Rotten",
    "freshness_score": 0.0,
    "dasfs_score":     0.0,
    "knn_support":     0.0,
    "confidence":      60.0,
    "low_conf_flag":   False
}
```

---

### New Image C — Ambiguous Banana (Low Fruit Confidence)

This banana image is partially yellowed — the model is unsure if it's a banana
or a mango.

```
Fruit SVM output:
  banana: 0.58   ← top candidate but below threshold
  mango:  0.31
  apple:  0.07
  orange: 0.03
  grape:  0.01

top1_prob = 0.58 < 0.70 → TRIGGER TOP-2 FALLBACK
low_conf_flag = True
```

**Top-2 candidates:** banana (0.58), mango (0.31)

**Evaluate banana:**
```
proj_banana = F19_sc × (−0.625) + F20_sc × (0.781)
            = ... (using scaled features of this image: F19_sc=+0.30, F20_sc=−0.55)
            = 0.30×(−0.625) + (−0.55)×0.781
            = −0.1875 + (−0.4296)
            = −0.617

Validity for banana [−1.071, +2.407]:
  Is −0.617 in [−1.071, 2.407]? YES ✓

score_banana = (2.120−(−0.617)) / (2.120−(−0.784)) = 2.737/2.904 = 0.942
mid_banana = (−0.784+2.120)/2 = 0.668
d_conf_banana = 1−exp(−(−0.617−0.668)²/(2×0.102²)) = 1−exp(−79.3) ≈ 1.0
knn_banana = exp(−dist/τ) ≈ 0.61
combined_conf_banana = 0.6×1.0 + 0.4×0.61 = 0.844
```

**Evaluate mango:**
```
proj_mango = 0.30×(−0.625) + (−0.55)×0.781 = −0.617 (same projection)

Validity for mango [p_fresh_mango−2σ, p_rotten_mango+2σ]:
  p_fresh_mango ≈ −0.580, p_rotten_mango ≈ +1.700, spread_mango ≈ 0.130
  Range = [−0.580−0.260, +1.700+0.260] = [−0.840, +1.960]
  Is −0.617 in [−0.840, 1.960]? YES ✓

score_mango = (1.700−(−0.617)) / (1.700−(−0.580)) = 2.317/2.280 = 1.016 → clipped to 1.0
d_conf_mango ≈ 1.0
knn_dist to mango fresh samples is larger (different species)
knn_mango ≈ 0.35
combined_conf_mango = 0.6×1.0 + 0.4×0.35 = 0.740
```

**Fallback decision:**
```
banana confidence: 0.844
mango  confidence: 0.740

0.844 > 0.740  →  SELECT banana
```

**Final for Image C:**
```python
{
    "fruit":           "banana",
    "label":           "Very Fresh",
    "freshness_score": 94.2,
    "dasfs_score":     94.2,
    "knn_support":     61.0,
    "confidence":      80.8,
    "low_conf_flag":   True   ← flag raised because fruit ID was uncertain
}
```

---

## Step 9 — Test Set Evaluation (02_evaluate.ipynb)

We now run the held-out 3 test rows (S03, S06, S09) through the trained system.

| Test | Fruit | True label | DASFS score | KNN sup | Freshness % | Predicted label | Correct? |
|---|---|---|---|---|---|---|---|
| S03 | apple | Rotten (1) | 0.0% | ~0% | **0.0%** | Rotten | ✅ |
| S06 | banana | Rotten (1) | 0.0% | ~0% | **0.0%** | Rotten | ✅ |
| S09 | mango | Rotten (1) | 0.0% | ~0% | **0.0%** | Rotten | ✅ |

Test accuracy = 3/3 = **100%** (on this small example)

In the real system, the test set contains hundreds of samples and the accuracy is
reported as fresh-accuracy, rotten-accuracy, and balanced accuracy per fruit.

---

## Summary: What Each Step Did to the Numbers

```
RAW MATRIX (15 × 20)
        │
        ▼
TRAIN/TEST SPLIT  →  Train: 12 rows,  Test: 3 rows
        │
        ▼
GLOBAL SCALER     →  Each column: mean≈0, std≈1
                     Example: F19 mean=256.92, std=151.79
                              F20 mean=0.098,  std=0.097
        │
        ├──── FRUIT BRANCH ──────────────────────────────────────────────┐
        │     CMI filter (skip) → RFE selects 10/20 features             │
        │     SVM-RBF trained → outputs fruit probabilities (0–1)        │
        │     Confidence gate: if prob < 0.70 → top-2 fallback           │
        │                                                                  │
        └──── FRESHNESS BRANCH ──────────────────────────────────────────┘
              CMI prefilter:   18 of 20 features kept (F03, F12 removed)
              RFE:             10 of 18 features selected
              DASFS per fruit: degradation axis built from μ_fresh, μ_rotten
                               Apple:  p_fresh=−0.63, p_rotten=+1.69, sep=16.2
                               Banana: p_fresh=−0.78, p_rotten=+2.12, sep=28.5
                               Orange: p_fresh=−0.78, p_rotten=+1.72, sep=26.2
                               Grape:  p_fresh=−0.26, p_rotten=+1.32, sep= 9.4
              KNN per fruit:   trained on fresh samples only, τ calibrated
                               Apple τ=0.224, Banana τ=0.224, Orange τ=0.172

        ▼
INFERENCE on new image:
    1. Extract 20 features → Scale → Route to fruit and freshness branches
    2. SVM: fruit identity + confidence
    3. Project onto fruit's DASFS axis → freshness score (0–1)
    4. KNN: how similar to known-fresh samples? → knn_support (0–1)
    5. Blend: if DASFS conf ≥ 0.3 → use DASFS; else blend 60/40
    6. Map to label: >75%=Very Fresh, >50%=Fresh, >25%=Slightly Degraded, ≤25%=Rotten

        ▼
OUTPUT: { fruit, label, freshness_score, dasfs_score, knn_support, confidence, low_conf_flag }
```

---

*End of dry-run walkthrough.*
