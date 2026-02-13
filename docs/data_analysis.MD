# PTM Site Data Analysis Report

What follows is a runthrough of everything we did in `src/data.ipynb` for the PTM site prediction work. The notebook pulls in five PTM types (train and test), runs them through a set of checks and summaries, and turns the results into the tables and figures below. Use this doc as the single reference for what was analyzed and what came out of it.

---

## Dataset Overview

We work with five PTM types, each with its own train/test CSVs in `datasets/`:

| PTM Type | Modification   | Residue        |
|----------|----------------|----------------|
| acet_k   | Acetylation    | Lysine (K)     |
| crot_k   | Crotonylation  | Lysine (K)     |
| gly_n    | Glycosylation  | Asparagine (N) |
| phos_y   | Phosphorylation| Tyrosine (Y)   |
| sumo_k   | SUMOylation    | Lysine (K)     |

Each row is a sequence window; the label is binary (1 = PTM site at the target position, 0 = not). Files are named like `acet_k_train.csv` and `acet_k_test.csv`.

---

## 1. Data Loading and Initial Exploration

We load all ten files (train + test per PTM type) and inspect schema. All load clean. Sizes are in the 14k–71k range for train and 1.7k–8.6k for test depending on PTM type.

Two schemas show up. acet_k, gly_n, phos_y, and sumo_k share a **standard** layout: 13 columns, including `original_sequence`, `ptm_type`, `UniProt_ID`, the usual embedding/entropy columns (`embedding_dispersion`, `center_window_dispersion`, `embedding_entropy`, `center_window_entropy`, `svd_entropy`, `center_svd_entropy`), plus `cluster`, `cluster_label`, `LCR_prob`, and `HCR_prob`. No missing values in what we checked. crot_k is different: only four columns (`UniProt_ID`, `original_sequence`, `ptm_type`, `full_sequence`). No embedding, entropy, or cluster/probability fields there, so any analysis that relies on those features skips crot_k.

---

## 2. Dataset Statistics and Overview

Per-dataset we compute sample count, feature count, unique proteins, PTM vs non-PTM counts, imbalance ratio, and sequence length stats (mean, std, min, max) wherever we have `original_sequence`. That becomes the main summary table (`summary_df`) that feeds later sections and the exported CSVs.

---

## 3. Class Imbalance Analysis

PTM sites are rare. We combine train and test per PTM type, count positives and negatives, and plot pie charts (one per PTM type) and build a small table of imbalance ratios (non-PTM : PTM). The picture is the same everywhere: heavy skew toward non-PTM. That’s expected and is why we care about class weights, focal loss, resampling, and metrics like AUPRC in Milestone 1.

**Output:** [data_analysis/ptm_class_distribution.png](data_analysis/ptm_class_distribution.png)

---

## 4. Sequence Length Analysis

We take the length of each `original_sequence`, then mean, median, std, min, max, and quartiles per PTM type. Histograms with a mean line go into one figure. Lengths are fairly consistent within and across PTM types, which keeps the window-based setup comparable.

**Outputs:** [data_analysis/sequence_length_distribution.png](data_analysis/sequence_length_distribution.png), and the sequence-length stats table (`seq_length_df`).

---

## 5. Feature Analysis (Embedding and Entropy)

For the four standard-format PTM types we look at the six numeric features (embedding/center dispersion and entropy, plus SVD entropy). We summarize them (mean, std, median) overall and split by PTM vs non-PTM, then box-plot each feature by PTM type with two boxes per type (PTM vs non-PTM). The distributions do separate in places—enough to suggest these features carry signal. crot_k is left out here because it doesn’t have those columns.

**Outputs:** [data_analysis/feature_distributions.png](data_analysis/feature_distributions.png), plus the feature-stats table.

---

## 6. Cluster and Probability Analysis

On standard-format data we count HCR vs LCR from `cluster_label` and average `LCR_prob` and `HCR_prob` per PTM type. Gives a quick read on sequence complexity and on the existing predictor outputs we can use as baselines or extra features.

---

## 7. Protein-Level Analysis (UniProt ID Distribution)

We need to know how many proteins we have and how many windows per protein. Per PTM type we get unique UniProt_ID count, total samples, average samples per protein, and (where possible) average and max PTM sites per protein. Two bar charts: one for unique proteins, one for total samples. The takeaway: many windows per protein, i.e. sliding windows. So we must split by protein, not by row, and use group-aware CV if we do cross-validation.

**Outputs:** [data_analysis/protein_distribution.png](data_analysis/protein_distribution.png), protein stats table (`protein_df`).

---

## 8. Cross-PTM Type Comparison

One table per PTM type (train+test combined): total samples, unique proteins, PTM vs non-PTM counts, PTM share, mean sequence length. Used in the long summary and exported as `ptm_comparison.csv`.

---

## 9. Key Insights and Trends Summary

A short narrative that ties together imbalance, dataset sizes, sequence length behavior, and the protein-level point: we need protein-wise splits and we have a clear size/complexity picture before building the MLP baseline.

---

## 10. Amino Acid Composition Analysis

We ask whether PTM windows look different from non-PTM windows in terms of amino acid usage. For each standard-format PTM type we compute per-residue frequencies in PTM vs non-PTM sequences, take the difference (PTM − non-PTM), and report the top enriched and top depleted residues. Bar charts compare frequencies side by side. The patterns differ by PTM type—good justification for composition or k-mer features and for PTM-specific modeling.

**Outputs:** [data_analysis/amino_acid_composition.png](data_analysis/amino_acid_composition.png), composition table exported as `data_analysis/amino_acid_composition.csv`.

---

## 11. Position-Specific Pattern Analysis

For PTM sites only we aggregate amino acid counts by offset from the window center (e.g. ±5). We summarize the most frequent residue at center and at a few flanking positions per PTM type. The center usually matches the modified residue (K for lysine PTMs, Y for phos_y, N for gly_n). Flanking preferences are there too and can feed into local-context or motif features.

---

## 12. Statistical Significance Testing

We test whether the embedding and entropy features actually differ between PTM and non-PTM sites. For each standard-format PTM type and each of the six numeric features we run a two-sided Mann-Whitney U test, store the statistic, p-value, mean difference, and a yes/no for p < 0.05, and then summarize how many features are significant per type. Plenty of them are, which backs using these features in the MLP and beyond.

**Output:** `statistical_tests.csv` in `data_analysis/`.

---

## 13. Feature Correlation Analysis

We correlate each numeric feature (including LCR_prob and HCR_prob when present) with the binary `ptm_type` label per PTM type. We report the top correlated features and the max absolute correlation. Handy for feature selection and for comparing against simple baselines.

**Output:** `feature_correlations.csv` in `data_analysis/`.

---

## 14. Comprehensive Summary and Key Insights

The notebook’s final summary block is a long narrative that hits: dataset sizes (who’s biggest/smallest), class imbalance and what to do about it, sequence length and window consistency, amino acid composition takeaways, which features are significant and correlated, protein-level implications, PTM-specific notes (lysine vs tyrosine vs asparagine; crot_k’s different format), data quality, and a set of recommendations for model development. It also lists the main trends (imbalance, consistent windows, distinct AA preferences, discriminative features, need for protein-wise splitting). All of that is in the notebook; this report points to it rather than duplicating it line for line.

---

## Recommendations for Model Development

The notebook folds these into the comprehensive summary. In short:

- **Imbalance:** Use class weights, focal loss, or resampling; lean on precision-recall and AUPRC.
- **Splitting:** Protein-wise train/validation/test only; group-based CV when needed.
- **Features:** Use the existing embedding/entropy and composition (or k-mer) features; trim with feature selection if useful.
- **Architecture:** Start with an MLP on fixed-window features (Milestone 1), then move to sequence models (e.g. ProtMAMBA).
- **Metrics:** AUPRC first; then precision, recall, F1 at a chosen threshold; per-protein evaluation where it makes sense.
- **Baselines:** Compare against LCR_prob/HCR_prob and record gains at each step.

---

## Exports

All data-analysis outputs (figures and CSVs) live under `docs/data_analysis/`. When you run `src/data.ipynb`, configure it to write there so the following assets stay in one place.

**CSVs:** `dataset_summary.csv`, `class_imbalance.csv`, `sequence_length_stats.csv`, `protein_statistics.csv`, `ptm_comparison.csv`, `amino_acid_composition.csv`, `statistical_tests.csv`, `feature_correlations.csv`.

**Figures:** `ptm_class_distribution.png`, `sequence_length_distribution.png`, `feature_distributions.png`, `protein_distribution.png`, `amino_acid_composition.png`.

---

## Available figures (documentation)

The following figures are produced by the data-analysis notebook and are stored in `data_analysis/` for use in this report.

| Figure | Description |
|--------|-------------|
| [ptm_class_distribution.png](data_analysis/ptm_class_distribution.png) | Class imbalance (PTM vs non-PTM) per PTM type; pie charts. |
| [sequence_length_distribution.png](data_analysis/sequence_length_distribution.png) | Distribution of sequence window lengths by PTM type; histograms with mean line. |
| [feature_distributions.png](data_analysis/feature_distributions.png) | Box plots of embedding/entropy features by PTM type and label (PTM vs non-PTM). |
| [protein_distribution.png](data_analysis/protein_distribution.png) | Bar charts: unique proteins and total samples per PTM type. |
| [amino_acid_composition.png](data_analysis/amino_acid_composition.png) | Amino acid frequency comparison (PTM vs non-PTM windows) per PTM type. |

---

## Quick Reference

| Section | What it does | Main outputs |
|--------|----------------|--------------|
| 1 | Load and inspect all PTM datasets | Row/column counts, schema comparison |
| 2 | Per-dataset summary stats | summary_df |
| 3 | Class imbalance | Pie charts, imbalance table, ptm_class_distribution.png |
| 4 | Sequence length | Histograms, seq_length_df, sequence_length_distribution.png |
| 5 | Embedding/entropy features | Feature stats, box plots, feature_distributions.png |
| 6 | Clusters and precomputed probs | cluster_df (HCR/LCR, LCR_prob, HCR_prob) |
| 7 | Protein-level counts | protein_df, protein_distribution.png |
| 8 | Cross-PTM comparison | comparison_df, ptm_comparison.csv |
| 9 | Short insight summary | Text summary of 2–8 |
| 10 | Amino acid composition | aa_comp_df, amino_acid_composition.png/.csv |
| 11 | Position-specific patterns | Center and flanking AA summary |
| 12 | Mann-Whitney U on features | statistical_tests.csv |
| 13 | Feature–label correlations | feature_correlations.csv |
| 14 | Full narrative + recommendations | In-notebook only |
| — | Export | All CSVs and figures in data_analysis/ |

This report and the notebook together cover what’s needed for Milestone 1: dataset choice, label setup, protein-wise splits, imbalance strategy, and baseline model design.
