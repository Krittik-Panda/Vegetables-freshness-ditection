# Fruit Freshness Detection — Explained Simply

> **What does this system do?**
> You take a photo of a fruit. This system looks at the photo and tells you two things:
> 1. What fruit is it? (apple, banana, mango, etc.)
> 2. How fresh is it? (0–100%, with a label like "Very Fresh" or "Rotten")
>
> It does all of this automatically, without any human looking at the photo.

---

## The Big Picture — Four Steps

Think of the pipeline like a factory assembly line with four stations:

```
Photos  →  [Station 1: Read & Describe]  →  [Station 2: Learn]  →  [Station 3: Check]  →  [Station 4: Use]
```

| Station | Notebook | What happens |
|---|---|---|
| 1 | `00_extract` | Every photo is described as a list of numbers |
| 2 | `01_train` | The computer studies those numbers and learns patterns |
| 3 | `02_evaluate` | We test how well the computer learned |
| 4 | `03_predict` | We feed new photos and get freshness scores |

---

## Station 1 — Describing Every Photo as Numbers (`00_extract.ipynb`)

Computers cannot "see" a photo the way humans do. Instead, they need every photo
converted into a list of numbers that captures what the photo looks like. This
station does exactly that.

---

### Step 0 — The Smart Cache (Dataset Fingerprint)

**What the code does:**
Before doing any work, the program checks whether the photos have changed since
the last time it ran. It does this by creating a unique "fingerprint" (called an
MD5 hash) of all the photo files — basically a short code that summarises the
entire photo collection.

**In plain English:**
Imagine you have a huge pile of books. Instead of re-reading all of them every
time someone asks "have any books changed?", you weigh the whole pile. If the
weight is the same as last time, nothing changed — skip the work. If the weight
is different, something is new.

**Why it matters:**
Converting thousands of photos into numbers takes several minutes. If nothing
has changed, the system loads the saved results instantly instead of redoing all
that work. This alone saves 5–10 minutes every time you re-run the notebook
without adding new images.

---

### Step 1 — Reading and Resizing Photos

**What the code does:**
All photos are read from disk and resized to exactly 224×224 pixels. Multiple
photos are read at the same time using parallel threads to speed things up.

**In plain English:**
Think of it like a document scanner that resizes everything to A4 before
scanning. No matter if the original photo was taken on a phone or a DSLR — it
all becomes the same size so the system can compare them fairly.

---

### Step 2a — The "Human Eye" Features (30 numbers per photo)

**What the code does:**
For each photo, 30 hand-designed measurements are taken. These are grouped into
five categories:

| Category | How many numbers | What it measures |
|---|---|---|
| **Colour (RGB)** | 6 | Average and variation of red, green, and blue |
| **Colour (HSV)** | 6 | Hue (what colour?), saturation (how vivid?), brightness — in a way that handles colour wrapping correctly |
| **Colour (LAB)** | 6 | Lightness and two colour dimensions that match how humans perceive colour differences |
| **Texture** | 5 | How sharp the image is, surface pattern regularity, and how "complex" the pixel intensities are |
| **Shape** | 6 | The outline of the fruit — how big, how round, how convex, its aspect ratio |
| **Darkness** | 1 | What fraction of the photo is very dark — a strong rot indicator |

**In plain English:**
A trained fruit inspector at a market uses similar cues without thinking about
it. They notice: "This mango has lost its bright orange colour (colour shift),
the skin looks wrinkled (texture change), there's a dark bruise patch (darkness)".
We are just writing those observations down as numbers.

---

### Step 2b — The "Deep AI" Features (1280 numbers per photo)

**What the code does:**
A large pretrained neural network called **EfficientNetB0** — trained by Google
on 1.2 million photos — looks at each photo and produces a list of 1280 numbers
that capture high-level patterns it learned from those millions of images.

**In plain English:**
Imagine hiring someone who has seen millions of objects and fruits in their life.
They can look at a photo and instantly summarise it in a way that captures subtle
patterns a beginner would miss — texture nuances, colour gradients, fine surface
details. The 1280 numbers are their "summary" of the photo.

The network runs photos in groups of 64 at a time (batching) to be efficient.

---

### Step 2c — Combining Both Descriptions (1310 numbers per photo)

**What the code does:**
The 30 hand-crafted numbers and the 1280 deep-AI numbers are simply joined
end-to-end into a single list of **1310 numbers** per photo. This combined list
is called `X_fused`.

**In plain English:**
You have two expert opinions — a human inspector (30 numbers) and an AI that
has studied millions of photos (1280 numbers). You write both opinions on the
same sheet of paper so the learning stage can use both.

---

### Saved Outputs

After this station finishes, four files are saved:

| File | What it contains |
|---|---|
| `X_fused.npy` | A giant table — one row per photo, 1310 columns of numbers |
| `y.npy` | A list saying "0 = fresh, 1 = rotten" for each photo |
| `fruit_type.npy` | A list saying which fruit species each photo belongs to |
| `dataset_hash.txt` | The fingerprint so next time we know if photos changed |

---

## Station 2 — Learning Patterns from the Numbers (`01_train.ipynb`)

Now the system studies the 1310-number descriptions of all the labelled photos
and builds models — mathematical rules — that can identify fruit type and
freshness in new, unseen photos.

---

### Step A — Safety Check

**What the code does:**
Before anything, the notebook checks that `X_fused.npy` actually exists and that
its fingerprint matches the current photos. If not, it stops with a clear error
message.

**In plain English:**
Like a chef who checks the ingredients are fresh before starting to cook. No
point proceeding if the data is stale or missing.

---

### Step B — Cleaning Fruit Names

**What the code does:**
The dataset folders are named things like `"freshapple"` and `"rottenapple"`.
The code strips the `fresh` / `rotten` prefix so that `fruit_type` contains
just `"apple"`. The freshness label is already stored separately in `y.npy`.

**In plain English:**
The folder names are redundant — they shout both the fruit name AND freshness
status. We only need the fruit name here (freshness is already recorded
elsewhere), so we trim off the prefix.

**Why it matters:**
Without this fix, the system sees `"freshapple"` and `"rottenapple"` as two
completely different fruits instead of two freshness states of the same fruit.
The DASFS scoring (explained below) would get zero rotten samples per fruit —
making it completely broken.

---

### Step C — Splitting into Training and Test Sets

**What the code does:**
80% of the photos are used for training (the system learns from these), and 20%
are held back for testing (the system is tested on these as if seeing them for
the first time). The split is done carefully so that every fruit–freshness
combination is proportionally represented in both halves.

**In plain English:**
Imagine you have 1000 sample exam questions. You study 800 of them. Then you sit
an exam using the remaining 200 questions — ones you never saw during studying.
The held-back 200 questions measure real performance, not just memorisation.

The "careful proportional split" means: if 10% of photos are rotten bananas,
then 10% of the training set and 10% of the test set are also rotten bananas.
No fruit gets accidentally left out of either half.

---

### Step D — Standardising the Numbers (Global Scaling)

**What the code does:**
Each of the 1310 number columns is re-scaled so it has a mean of 0 and a
standard deviation of 1. The scaling formula is calculated from the training
data only, then applied to both training and test data.

**In plain English:**
Imagine one column measures "fruit size in millimetres" (values: 40–200) and
another measures "redness fraction" (values: 0–1). These are on completely
different scales. If you fed both directly into a learning algorithm, the large
millimetre numbers would drown out the small fractions. Scaling fixes this so
every column has equal weight.

Crucially, the scaling formula is calculated from training photos only —
applying the test-set values to calculate the formula would be "peeking at the
exam answers."

---

### Step E — Fruit Branch: Which Fruit Is This?

This branch builds a model that looks at a photo's numbers and decides what
fruit it is.

#### Feature Selection — Picking the Most Useful Numbers

**What the code does:**
Out of 1310 numbers, not all are equally useful for identifying the fruit type.
A method called **Recursive Feature Elimination (RFE)** is used to
automatically find which subset of numbers gives the best fruit identification
accuracy. It tests subsets of 80, 100, 120, and 140 features using 5-fold
cross-validation, running all five test-folds simultaneously across CPU cores
for speed.

**In plain English:**
Imagine you have 1310 clues to identify a suspect. Some clues are useful (shirt
colour, height), some are noise (shoe brand in a dark photo). RFE methodically
discards the least-useful clues until it finds the smallest set that still gives
accurate identification.

"5-fold cross-validation" means: divide the training data into 5 chunks. Train
on 4 chunks, test on the 5th. Repeat for each chunk. Average the results. This
gives a reliable estimate without needing extra data.

#### The Fruit Classifier

**What the code does:**
Once the best features are selected, a **Support Vector Machine (SVM) with an
RBF kernel** is trained. This specific model (`C=10, probability=True`) can
output not just "which fruit" but also a *confidence percentage* — which is
critical for the next step.

**In plain English:**
An SVM draws mathematical boundaries between categories in a high-dimensional
space. Think of it as drawing curved fence lines around clusters of "apple
points", "banana points", "mango points" etc. A new photo gets classified based
on which fenced region it falls into. `probability=True` means it also tells you
"I'm 85% sure this is an apple, 10% sure it's a pear."

---

### Step F — Freshness Branch: Selecting the Right Numbers for Freshness

#### CMI Prefilter — Finding Freshness-Relevant Features

**What the code does:**
A statistical measure called **Conditional Mutual Information (CMI)** scores
each of the 1310 numbers by how much information it carries about freshness,
*specifically within each fruit species*. The top 300 scoring features are kept.

The formula is:
```
CMI score for feature i = sum over each fruit of:
    (fraction of data that is this fruit) × (how much feature i predicts freshness for this fruit)
```

**In plain English:**
Redness might be a great indicator of freshness in strawberries but useless for
bananas (which go brown, not pale). CMI finds the features that are genuinely
informative about freshness *given which fruit we're looking at*, not just
features that happen to correlate with freshness in general.

This step cuts from 1310 features down to 300, dramatically reducing the work
for the next step.

#### RFE Feature Selection for Freshness

Same cross-validated RFE process as the fruit branch, but now running on the
300 CMI-selected features, and now aiming to predict freshness (fresh vs rotten)
rather than fruit type.

---

### Step G — DASFS: Building a "Freshness Ruler" Per Fruit

This is the most unique and important part of the system. DASFS stands for
**Dual-Anchor Spectral Freshness Scoring**.

**What the code does:**
For each fruit species, the system does the following using the freshness feature
space:

1. Compute the "average fresh version" of that fruit — the centroid of all fresh
   training samples
2. Compute the "average rotten version" of that fruit — the centroid of all
   rotten training samples
3. Draw an imaginary line from average-fresh to average-rotten. This is the
   **degradation axis** — the direction in feature space that most clearly
   separates freshness from rottenness for this specific fruit
4. Record two anchor points on that line: `p_fresh` (where fresh fruits
   typically land) and `p_rotten` (where rotten fruits typically land)
5. Record the `spread` — how spread out fresh and rotten samples are around
   their anchors

**In plain English:**
Imagine plotting every photo of an apple as a dot in space, where the position
of each dot is determined by its 1310 numbers. Fresh apples cluster in one
region; rotten apples cluster in another. DASFS draws a straight line between
the two clusters and marks the "typical fresh" and "typical rotten" positions on
that line. This line becomes a **ruler** — you can place any new apple on it and
read off how fresh it is.

Critically, **every fruit gets its own ruler**. A banana's freshness ruler is
completely different from an apple's, because they rot differently.

**Output saved:** `dasfs.pkl` — one ruler (axis + anchor points) per fruit.

---

### Step H — KNN: A "Nearest Neighbours" Freshness Backup

**What the code does:**
For each fruit, a **K-Nearest Neighbours (KNN)** model is trained using only
fresh training samples. For any new photo, the model finds the 5 most similar
fresh photos in the training set and reports how similar the new photo is to
them.

The similarity is expressed as:
```
KNN support = exp(−distance / τ)
```
where `τ` is the typical distance between fresh samples of that fruit. A score
near 1.0 means "very similar to known-fresh samples"; near 0.0 means "unlike
anything we've seen as fresh."

**In plain English:**
Think of it as asking 5 trusted fresh-fruit experts: "Does this fruit look
similar to anything you've seen that was definitely fresh?" If all 5 say yes,
the KNN support is high. If they all say "this looks nothing like a fresh
[fruit name]", the support is low.

This acts as a sanity check on the DASFS ruler — if the DASFS score is
uncertain, the KNN vote helps decide.

**Output saved:** `knn_dict.pkl` — one KNN model + threshold per fruit.

---

## Station 3 — Testing How Well the System Learned (`02_evaluate.ipynb`)

Before deploying the system, we measure its accuracy on the held-back 20% of
photos it has never seen. Six diagnostic charts are produced.

---

### Plot 1 — Did We Pick the Right Number of Features?

Shows the 5-fold cross-validation accuracy for each candidate feature count (80,
100, 120, 140) for both the fruit and freshness branches. The best count is
highlighted in red.

**In plain English:**
Did using 100 features work better than 80? Better than 140? This chart answers
that question so you can see whether the chosen count was sensible.

---

### Plot 2 — Fruit Confusion Matrix

Shows exactly which fruit types get confused with each other. The rows are the
actual fruit, the columns are what the model predicted. Diagonal entries are
correct predictions; off-diagonal entries are mistakes.

**In plain English:**
Did the model ever call a mango a papaya? This table shows all such confusions
at a glance. A perfect model would have numbers only on the diagonal.

---

### Plot 3 — Per-Fruit Freshness Accuracy

For each fruit, three bars are shown:
- **Green** — of the truly fresh samples, what fraction did the model correctly
  call fresh?
- **Red** — of the truly rotten samples, what fraction did the model correctly
  call rotten?
- **Blue** — the balanced average of the two

**In plain English:**
The system might be great at detecting rotten apples but rubbish at detecting
rotten bananas. This chart reveals those per-fruit weaknesses so you know where
to improve.

Note: this uses the ground-truth fruit labels, not the model's predicted labels,
so it only measures freshness accuracy without contamination from fruit
misidentification errors.

---

### Plot 4 — Overall Freshness Accuracy

The same idea but aggregated across all fruits. Reports:
- Overall accuracy
- Balanced accuracy (accounts for unequal fresh/rotten sample counts)
- Precision, recall, and F1 for both Fresh and Rotten classes

---

### Plot 5 — KNN Distance Distributions

For each fruit, shows two overlapping histograms: the KNN distances for fresh
samples (green) and rotten samples (red), with the threshold τ marked.

**In plain English:**
Fresh samples should cluster near known-fresh photos (small distance). Rotten
samples should be further away. If the green and red histograms overlap heavily,
the KNN model is not distinguishing well for that fruit.

---

### Plot 6 — DASFS Projection Distributions

For each fruit, shows the distribution of where fresh and rotten samples land
on the DASFS freshness ruler, with the `p_fresh` and `p_rotten` anchor points
marked.

**In plain English:**
Is the "freshness ruler" actually working? If fresh apples all land near
`p_fresh` and rotten apples all land near `p_rotten` with a clear gap between
them, the ruler is working. If the two distributions heavily overlap, the
DASFS axis is not very discriminative for that fruit.

---

## Station 4 — Using the System on New Photos (`03_predict.ipynb`)

This is the station you use in production. You give it a photo; it gives you a
freshness report.

---

### Step A — Load All Models (Once Per Session)

All 8 saved model files plus the EfficientNetB0 backbone are loaded into memory
when the notebook starts. This takes ~20 seconds and only happens once — all
subsequent predictions are fast.

---

### Step B — Extract Features from the New Photo

The exact same 1310-number extraction process from Station 1 is applied to the
new photo: EfficientNetB0 (1280 numbers) + handcrafted (30 numbers) = 1310
numbers total.

---

### Step C — Route Through Saved Transforms

The 1310 raw numbers are passed through the saved scaling and feature-selection
transformations:

```
1310 raw numbers
    → Global scaler (standardise all 1310)
         ├── Fruit RFE + Fruit Scaler  → best_N fruit features
         └── Top-300 CMI filter → Freshness RFE + Fresh Scaler  → best_N fresh features
```

The two resulting feature vectors go to different models.

---

### Step D — Fruit Identification with Confidence Gate

**What the code does:**
The SVM fruit classifier outputs a probability for each known fruit. If the
top probability is **70% or higher**, that fruit is accepted as the answer. If
it is below 70%, the system knows it is uncertain and triggers the fallback.

**In plain English:**
If the model says "I'm 85% sure this is an apple", it proceeds with apple. But
if it says "I'm only 55% sure — could be a mango or a papaya", it runs extra
checks instead of guessing blindly. Guessing the wrong fruit would feed the
freshness score into the wrong ruler, producing a meaningless result.

---

### Step E — Top-2 Fallback (When Confidence is Low)

**What the code does:**
When confidence is below 70%, the system evaluates both the top-1 and top-2
candidate fruits. For each:

1. Compute a DASFS freshness score and confidence
2. Check whether the photo's position on the freshness ruler falls within a
   reasonable range for that fruit (the validity check, see below)
3. Pick whichever candidate produces a more confident, valid result

If neither candidate passes the validity check, fall back to the top-1 guess.

**In plain English:**
"I'm not sure if this is a mango or papaya. Let me check — does it make sense
as a mango on the mango freshness ruler? Does it make sense as a papaya on the
papaya ruler? Use whichever fruit's ruler gives a cleaner answer."

---

### Step F — The DASFS Validity Check (Bug Fix)

**What the code does:**
Before using a DASFS score, the system checks that the photo's position on the
freshness ruler falls within a reasonable range:

```
Valid range: from (p_fresh − 2 × spread) to (p_rotten + 2 × spread)
```

An older version of this check incorrectly rejected photos that were extremely
fresh (they projected far towards the "very fresh" end and triggered a false
alarm). The corrected check uses the full expected range instead.

**In plain English:**
The ruler has a "valid measurement zone" slightly beyond both anchor points. If
a photo's measurement falls completely outside this zone, the ruler probably
doesn't apply to this fruit — either it's the wrong fruit, or something is
wrong. The old check was like a ruler that rejected measurements at its own
ends; the fix extends the valid zone to include the full expected range.

---

### Step G — Computing the Final Freshness Score

**What the code does:**

1. **DASFS score** — where on the ruler does this fruit sit?
   ```
   score = (p_rotten − projection) / (p_rotten − p_fresh)
   ```
   A score of 1.0 means "exactly at the fresh anchor"; 0.0 means "exactly at
   the rotten anchor".

2. **DASFS confidence** — how far is the projection from the midpoint of the
   ruler? Far from the midpoint = high confidence in the reading.

3. **KNN support** — how close is this photo to the nearest known-fresh
   examples?

4. **Blending the scores:**
   - If DASFS confidence is **high** (≥ 0.3): trust the DASFS score directly
   - If DASFS confidence is **low** (< 0.3): blend it with KNN support:
     `freshness = 0.6 × dasfs_score + 0.4 × knn_support`

5. **Overall confidence** (for transparency):
   `confidence = 0.6 × dasfs_confidence + 0.4 × knn_support`

**In plain English:**
The DASFS ruler gives the primary reading. The KNN "expert panel" provides a
backup vote. When the ruler reading is near the middle (uncertain), the expert
panel gets more weight. When the ruler reads clearly at one end (definite fresh
or definite rotten), the ruler is trusted on its own.

---

### Step H — Assigning a Freshness Label

The final score (0–100%) is converted to a human-readable label:

| Score | Label |
|---|---|
| Above 75% | **Very Fresh** — buy/eat immediately |
| 50–75% | **Fresh** — still good |
| 25–50% | **Slightly Degraded** — use soon |
| Below 25% | **Rotten** — discard |

---

### Step I — What You Get Back

```
{
  "fruit":            "apple",
  "label":            "Fresh",
  "freshness_score":  78.4,   ← main result, 0–100%
  "dasfs_score":      80.1,   ← what the ruler said
  "knn_support":      74.2,   ← what the expert panel said
  "confidence":       76.9,   ← how sure is the system?
  "low_conf_flag":    false   ← was the fruit ID uncertain?
}
```

---

### Step J — Batch Mode

Drop any number of photos into the `test_img/` folder and run the batch cell.
The system predicts all of them sequentially and outputs:
- A summary table with filename, fruit, label, score, and confidence
- A per-fruit breakdown showing average scores and label distribution
- A colour-coded bar chart saved as a PNG file

---

## All the Saved Files (Artifact Manifest)

```
pipeline_v2/
│
├── models/                          ← the trained brain of the system
│   ├── scaler.pkl                   numbers standardisation formula
│   ├── rfe_fruit.pkl                which features to use for fruit ID
│   ├── scaler_fruit.pkl             secondary scaling for fruit features
│   ├── svm_fruit.pkl                the fruit classifier
│   ├── rfe_fresh.pkl                which features to use for freshness
│   ├── scaler_fresh.pkl             secondary scaling for freshness features
│   ├── dasfs.pkl                    the freshness ruler for every fruit
│   └── knn_dict.pkl                 the nearest-neighbours backup models
│
└── artifacts/                       ← data and diagnostic files
    ├── X_fused.npy                  all photos as 1310-number rows
    ├── y.npy                        fresh/rotten labels
    ├── fruit_type.npy               fruit species per photo
    ├── dataset_hash.txt             fingerprint to detect dataset changes
    ├── X_fruit_train/test.npy       fruit-branch features, train and test
    ├── X_fresh_train/test.npy       freshness-branch features, train and test
    ├── y_train/test.npy             labels for each split
    ├── ft_train/test.npy            fruit types for each split
    ├── top300_cmi.npy               which 300 features matter for freshness
    ├── fresh_final_idx.npy          final freshness feature indices
    ├── cv_results.json              accuracy-vs-feature-count results
    └── plot_*.png                   all evaluation charts
```

---

## Common Questions

**Q: Why use both handcrafted features and a deep neural network?**
A: The neural network is excellent at capturing complex visual patterns but is
a "black box" — we don't know exactly what it is picking up. The handcrafted
features target known, interpretable freshness signals (colour, texture,
darkness). Combining both gives better accuracy than either alone.

**Q: Why does each fruit get its own freshness ruler (DASFS) and its own KNN model?**
A: Fruits rot differently. A ripe banana turns from yellow to brown (colour
shift). A rotten apple gets mushy patches (texture and shape change). A wilted
lettuce loses its green vibrancy (saturation drop). A single universal "fresh
vs rotten" model would conflate all these different degradation patterns.
Per-fruit models let each species be judged on its own terms.

**Q: What does the confidence score tell me?**
A: It tells you how certain the system is in its freshness reading. A
confidence of 90% means the photo clearly resembles known-fresh or known-rotten
training examples. A confidence of 40% means the photo fell in a murky middle
zone and the score should be treated with caution.

**Q: What does the `low_conf_flag` mean?**
A: It means the system was not sure which fruit it was looking at (below 70%
confidence for fruit identification). The freshness score is still produced
using the best available guess, but you should treat it with more scepticism.

**Q: Why are there bug-fix notes (CHANGE 4, 5, 6, PATCH A) in the code?**
A: The system went through several iterations. The most critical fixes were:
- **PATCH A** — Without stripping the `fresh`/`rotten` prefix from folder names,
  the DASFS rulers were built with zero rotten samples per fruit — completely
  broken
- **CHANGE 4** — The validity check was accidentally rejecting perfectly valid
  very-fresh fruit measurements
- **CHANGES 5 & 6** — Without the confidence gate, a misidentified fruit would
  silently produce a meaningless freshness score

---

*End of plain-language documentation.*
