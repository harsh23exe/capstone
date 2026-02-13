## Pipeline Iteration 1 – Phosphorylation (phos_y) Baseline

This document summarizes the first end-to-end baseline pipeline for phosphorylation (tyrosine) PTM site prediction, implemented in `src/simple_pipeline.ipynb`. The goal for this iteration was to validate the data pipeline, establish a reasonable MLP baseline, and understand how far simple window-based features can go on the phos_y task.

---

### Data and splitting

- **PTM type**: `phos_y` (phosphorylation on tyrosine).
- **Source files**: `datasets/phos_y_train.csv`, `datasets/phos_y_test.csv`.
- **Schema (train/test)**:
  - `original_sequence`: fixed-length sequence window (length 101).
  - `ptm_type`: binary label (1 = PTM at center residue, 0 = non-PTM).
  - `UniProt_ID`: protein identifier for protein-wise splitting.
  - Embedding / entropy features:
    - `embedding_dispersion`, `center_window_dispersion`
    - `embedding_entropy`, `center_window_entropy`
    - `svd_entropy`, `center_svd_entropy`
  - Additional fields (`cluster`, `cluster_label`, `LCR_prob`, `HCR_prob`) are present but not used as inputs in this iteration.

Key dataset stats from the notebook:

- **Train set shape**: `(58,721 rows, 13 columns)`
- **Test set shape**: `(5,814 rows, 13 columns)`
- **Train label distribution**: `{0: 48,327, 1: 10,394}`  
  Approx. **17–18% positives**, confirming class imbalance.
- **Unique proteins**:
  - Train: `4,390` proteins
  - Test: `471` proteins

Protein-wise splitting strategy:

- Start from `phos_y_train.csv` and split **by `UniProt_ID`**:
  - 80% of proteins → **training set**
  - 20% of proteins → **validation set**
- Hold `phos_y_test.csv` out as the **final test set**.

Resulting sample counts:

- **Train samples**: `46,628`
- **Validation samples**: `12,093`
- **Test samples**: `5,814`
- **Train proteins**: `3,512`
- **Validation proteins**: `878`

This guarantees no protein overlaps between train/val/test and matches the protein-wise evaluation philosophy from Milestone 1.

---

### Input representation

This iteration uses a straightforward **residue-centered window encoding**:

- **Window length**: 101 residues per example (taken directly from `original_sequence`).
- **Amino acid alphabet**:
  - 20 standard residues: `ACDEFGHIKLMNPQRSTVWY`
  - Plus one **unknown** symbol (`X`) mapped to index 20.
- **Encoding**:
  - Each residue is one-hot encoded over 21 channels.
  - Shape per sample: `101 × 21` → flattened to a vector of length `2,121`.
- **Numeric features**:
  - Concatenate the six precomputed features:
    - `embedding_dispersion`, `center_window_dispersion`
    - `embedding_entropy`, `center_window_entropy`
    - `svd_entropy`, `center_svd_entropy`
  - Final **feature dimension**: `2,127`.

Short version: each window becomes a fixed-length vector combining **local sequence pattern** (one-hot window) with **global numeric descriptors** (embedding/entropy features).

---

### Model and training setup

**Model**: a simple fully-connected MLP implemented in PyTorch.

- **Architecture**:
  - Input: `2,127`-dimensional feature vector.
  - Hidden layers:
    - Linear → 512 units → ReLU → Dropout(0.3)
    - Linear → 256 units → ReLU → Dropout(0.3)
  - Output: Linear → 1 logit (binary classification).
  - Sigmoid is applied only for metric computation.

- **Loss and imbalance handling**:
  - `BCEWithLogitsLoss` with `pos_weight` to counter class imbalance.
  - From the training labels:
    - Positives: `n_pos = 10,394`
    - Negatives: `n_neg = 46,628 - 10,394 = 36,234`
    - Computed positive weight:  
      `weight_pos ≈ n_neg / n_pos ≈ 4.61`
  - This effectively upweights positive examples in the loss.

- **Optimization**:
  - Optimizer: Adam
  - Learning rate: `1e-3`
  - Batch size: `256`
  - Epochs: `15`
  - Device: CPU or GPU depending on availability; the notebook handles this automatically.

- **Validation procedure**:
  - At the end of each epoch:
    - Compute probabilities on the validation loader.
    - Derive `precision`, `recall`, `F1`, `AUPRC` at threshold 0.5.
  - Track:
    - `train_loss` per epoch
    - `val_auprc`, `val_f1` per epoch
  - **Model selection**:
    - Keep the state dict corresponding to the **best validation AUPRC** observed during training.

---

### Validation behaviour

The validation metrics (protein-wise split) behave as follows over 15 epochs:

- **Train loss**:
  - Starts around ~1.00
  - Drops steadily to ~0.035 by epoch 15, indicating the MLP fits the window features well.

- **Validation AUPRC**:
  - Early improvement from ~0.42 at epoch 1 to a peak of **~0.49 around epoch 3**.
  - Afterwards it fluctuates in the **0.46–0.48** range, with a slight decline as the model continues to optimize the training loss.

- **Validation F1**:
  - Stabilizes in the **0.43–0.48** range.
  - The best epochs show a reasonable precision/recall trade-off but no dramatic gains beyond the first few epochs.

High-level takeaway: the model quickly captures the main signal in the fixed-window representation, then hits a plateau where further epochs mainly reduce training loss without meaningful validation gains. This is consistent with a relatively simple MLP capacity on a strongly imbalanced, noisy biological classification problem.

---

### Test set performance

The best model by validation AUPRC was evaluated on the held-out **phos_y test set** (`5,814` samples, `471` proteins).

At a fixed decision threshold of 0.5:

- **Precision**: `0.3789`
- **Recall**: `0.5773`
- **F1 score**: `0.4575`
- **AUPRC**: `0.4403`

Confusion matrix (rows = true label, columns = predicted label):

- `[[TN, FP], [FN, TP]] = [[3825, 967], [432, 590]]`

Interpretation:

- The model **recovers a majority of true PTM sites** (recall ~0.58) while maintaining **moderate precision** (~0.38).
- The **AUPRC of ~0.44** is clearly above a random baseline for this class balance and provides a meaningful reference point for future models.
- There is still a sizable number of **false positives** (967) and **false negatives** (432), which is expected for a first-pass window-based MLP.

The numbers above are for the **default** MLP config (512, 256, dropout 0.3, lr 1e-3, batch 256). A grid search over hyperparameters is reported in the next section.

---

### Qualitative error analysis

The notebook inspects **high-confidence errors** to build intuition:

- **False positives**:
  - Examples where the model assigns high probability to windows that are actually non-PTM.
  - Often look sequence-wise plausible (e.g., tyrosine at center with surrounding motifs similar to true sites), suggesting the MLP is picking up generic phosphorylation-like patterns that are not truly modified in the dataset.

- **False negatives**:
  - True PTM windows that the model scores low.
  - These often have weaker or atypical local patterns around the tyrosine, indicating that a pure fixed window plus global numeric features may miss subtler, long-range context.

Score distributions on the test set show:

- Non-PTM windows mostly cluster at low predicted probabilities, with a tail extending into higher scores.
- PTM windows are shifted toward higher probabilities but still overlap significantly with non-PTM windows.

This overlap is what limits achievable precision at reasonable recall levels and motivates richer models that can better separate the two distributions.

---

### Hyperparameter search

After the initial baseline, we ran a grid search over MLP hyperparameters via `src/hyperparameter_search.ipynb`. The idea was to see whether a different architecture or training setup could push validation (and test) performance up without changing the data or the feature pipeline.

**Search space:** 240 configurations in total. We varied:

- **Hidden dimensions:** (256, 128), (512, 256), (512, 256, 128), (1024, 512), (1024, 512, 256).
- **Dropout:** 0.2, 0.3, 0.4, 0.5.
- **Learning rate:** 1e-4, 5e-4, 1e-3, 2e-3.
- **Batch size:** 128, 256, 512.

Each configuration was trained for 15 epochs with the same protein-wise train/val split and the same class weighting. We selected the best run by **validation AUPRC** and then evaluated that model once on the held-out test set.

**Best configuration (run 20260205_152237):**

- **Hidden dims:** (1024, 512) — one step larger than the default (512, 256).
- **Dropout:** 0.2 (slightly lighter regularization than the baseline 0.3).
- **Learning rate:** 0.001 (same as baseline).
- **Batch size:** 128 (smaller than the baseline 256).

**Best metrics:**

- Validation AUPRC: **0.4902** (up from the baseline peak around 0.487).
- Test set (single evaluation): precision **0.3605**, recall **0.6360**, F1 **0.4602**, AUPRC **0.4390**.

So we get a small bump on validation AUPRC and a test F1 that’s marginally higher than the baseline (0.460 vs 0.458), with test AUPRC essentially unchanged. The best run trades a bit of precision for higher recall (0.64 vs 0.58), which is typical when you favor slightly larger capacity and lighter dropout. All 240 runs completed successfully; the full results and the best config are stored in `docs/` (timestamped CSV, JSON, and a short summary .txt). For iteration 2, we can lock in this config as the MLP baseline and focus on representation or architecture changes rather than more grid search on this same setup.

---

### What this baseline tells us

From this first iteration we can draw a few concrete conclusions:

- The **data pipeline is sound**:
  - Protein-wise split works and produces healthy train/val/test sizes.
  - Feature construction (one-hot window + numeric features) is stable and efficient.

- A **simple MLP on fixed-window features** already reaches:
  - Mid-0.4s **AUPRC** on test.
  - Balanced **recall vs precision** suitable for initial exploration and debugging.

- The model is **clearly capacity-limited**:
  - Training loss keeps dropping while validation AUPRC flattens and slightly degrades.
  - Local, non-contextual encoding is likely missing important sequence and protein-level context.

Overall, this is a solid Milestone 1 baseline: it proves the end-to-end PTM prediction loop works, surfaces realistic performance bounds for simple representations, and gives us concrete failure patterns to target in the next iterations.

---

### Next steps and ideas for iteration 2

A few directions that naturally follow from this baseline:

- **Threshold tuning and calibration**:
  - Explore operating points beyond the default 0.5 threshold, especially if precision at high recall is more valuable for downstream use.

- **Feature engineering**:
  - Add simple composition features, k-mer counts, or positional encodings around the center residue.
  - Experiment with including `LCR_prob`, `HCR_prob`, or cluster labels as additional inputs.

- **Modeling improvements**:
  - Replace or augment the MLP with a **sequence-aware architecture** (e.g., 1D convolutions or ProtMAMBA-style sequence models) that can capture longer-range context around the site.

- **Robustness checks**:
  - Per-protein performance summaries (e.g., distribution of AUPRC or F1 over proteins).
  - Stratified analysis across different sequence complexity or cluster labels.

These steps will build directly on the current notebook and use this iteration’s metrics as the baseline to beat.

