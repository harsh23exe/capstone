# Hyperparameter Search Across PTM Types

This document describes how hyperparameter search is run for each PTM type, where results are stored, and what the current outputs look like. The runner script and notebook are set up so each PTM’s artifacts stay in its own folder under `docs/`; only markdown lives at the top level.

---

## How it’s run

The script `src/run_hyperparameter_search_all_ptms.py` runs the search for every PTM type that uses the standard dataset schema: **acet_k**, **gly_n**, **phos_y**, **sumo_k**. (crot_k is skipped because its CSVs don’t have the embedding/entropy columns the pipeline expects.)

For each PTM type it:

1. Calls papermill to execute `src/hyperparameter_search.ipynb` with parameters `PTM_TYPE` and `MAX_CONFIGS=150`.
2. Writes the executed notebook to `docs/ptm_<PTM_TYPE>/execution.ipynb`.
3. The notebook itself writes all other outputs into the same folder: CSVs, best-config JSON, summary TXT, and the top-10 configs figure.

So one run per PTM fills exactly one folder. If you ever run the notebook manually multiple times (changing `PTM_TYPE` but not re-running the cell that sets `RESULTS_DIR`), all outputs can end up in the same folder. In that case, results were split by inferring PTM from metrics: phos_y has the lowest test AUPRC (~0.44), gly_n the highest (~0.98), and acet_k/sumo_k in between. The misplaced timestamped sets have been moved into their respective `ptm_<PTM_TYPE>/` folders.

---

## Where results live

| PTM type | Folder | Contents |
|----------|--------|----------|
| acet_k   | [docs/ptm_acet_k/](ptm_acet_k/) | execution notebook, one full result set (timestamp 20260208_013517) |
| gly_n    | [docs/ptm_gly_n/](ptm_gly_n/)   | execution notebook, one full result set (20260208_015008) |
| phos_y   | [docs/ptm_phos_y/](ptm_phos_y/) | execution notebook, three phos_y-only runs (20260205, 20260207, 20260208_024823) |
| sumo_k   | [docs/ptm_sumo_k/](ptm_sumo_k/) | execution notebook, one full result set (20260208_035950) |

Each run produces:

- **hyperparameter_search_results_&lt;timestamp&gt;.csv** – All configs with validation and test metrics (including MCC).
- **best_mlp_config_&lt;timestamp&gt;.json** – Best config by validation AUPRC and its test metrics (precision, recall, F1, AUPRC, MCC).
- **hyperparameter_search_summary_&lt;timestamp&gt;.txt** – Short human-readable summary of the best config and metrics.
- **top10_configs_&lt;timestamp&gt;.png** – Bar chart of validation AUPRC, test AUPRC, and test MCC for the top 10 configs.

---

## Current state: best run per PTM

Each PTM folder now contains that PTM’s own results. Below is the best configuration and test metrics from the latest run in each folder (by the timestamped files there).

**phos_y** (timestamp 20260208_024823) – Phosphorylation, tyrosine  
- Best config: (1024, 512), dropout 0.2, lr 0.001, batch 128  
- Val AUPRC 0.4902 | Test: P 0.3605, R 0.6360, F1 0.4602, AUPRC 0.4390, MCC 0.3254  

**acet_k** (timestamp 20260208_013517) – Acetylation, lysine  
- Best config: (1024, 512), dropout 0.5, lr 0.0005, batch 512  
- Val AUPRC 0.7721 | Test: P 0.6866, R 0.7193, F1 0.7026, AUPRC 0.7497, MCC 0.3782  

**gly_n** (timestamp 20260208_015008) – Glycosylation, asparagine  
- Best config: (512, 256, 128), dropout 0.3, lr 0.001, batch 128  
- Val AUPRC 0.9822 | Test: P 0.9500, R 0.9942, F1 0.9716, AUPRC 0.9790, MCC 0.9421  

**sumo_k** (timestamp 20260208_035950) – SUMOylation, lysine  
- Best config: (1024, 512), dropout 0.5, lr 0.0001, batch 256  
- Val AUPRC 0.7480 | Test: P 0.6180, R 0.7005, F1 0.6567, AUPRC 0.6836, MCC 0.3723  

Phos_y has three runs in its folder (two older 240-config runs plus 024823); the others have one run each. Use the JSON and CSV in each folder for full details.

---

## docs/ layout (markdown only at top level)

The top-level `docs/` directory is kept to markdown files only. All other assets live in subfolders:

- **docs/data_analysis/** – Data-analysis figures and CSVs from `src/data.ipynb`. Referenced by [data_analysis.MD](data_analysis.MD).
- **docs/ptm_acet_k/**, **docs/ptm_gly_n/**, **docs/ptm_phos_y/**, **docs/ptm_sumo_k/** – Hyperparameter search outputs per PTM type.

This keeps the main docs easy to scan and keeps images, configs, and tables grouped by purpose.
