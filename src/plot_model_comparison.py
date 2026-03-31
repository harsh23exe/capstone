"""Plot metric comparisons across models and PTMs.

Inputs:
- ptmgpt2 heldout metrics: embedded below
- baseline MLP metrics: embedded below
- CNN+GRU metrics: embedded below
- MultiClassifier metrics: embedded below

Outputs (written to ./docs/plots):
- model_comparison_metrics.csv
- comparison_<metric>.png
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Nordic research theme (user-provided base color)
# Base RGB: 0.357R, 0.537G, 0.482B
NORDIC_BASE_RGB = (0.357, 0.537, 0.482)

# Precomputed light/dark variants derived from NORDIC_BASE_RGB.
NORDIC_DARK_RGB = (0.26775, 0.40275, 0.36150)  # 75% base (blend 25% black)
NORDIC_LIGHT_RGB = (0.58205, 0.69905, 0.66330)  # blend 35% white
NORDIC_PALE_RGB = (0.77495, 0.83795, 0.81870)  # blend 65% white
NORDIC_EDGE_RGB = (0.16065, 0.24165, 0.21690)  # 45% base (blend 55% black)
NORDIC_GRID_RGB = (0.83925, 0.88425, 0.87050)  # blend 75% white


MODEL_COLORS = {
    "ptmgpt2 (heldout)": NORDIC_BASE_RGB,
    "baseline_mlp": NORDIC_LIGHT_RGB,
    "CNN+GRU": NORDIC_DARK_RGB,
    "MultiClassifier": NORDIC_PALE_RGB,
}


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": NORDIC_EDGE_RGB,
        "axes.labelcolor": "#111111",
        "text.color": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "grid.color": NORDIC_GRID_RGB,
        "grid.linewidth": 1.0,
        "axes.titleweight": "semibold",
    }
)


# PTMs to exclude from plots/exports
EXCLUDED_PTMS = {"gly_n", "phos_st"}


METRICS = ("mcc", "f1", "recall", "accuracy")


# Provided by user (ptmgpt2, heldout)
PTMGPT2_HELDOUT_RESULTS = {
    "acet_k": {"mcc": 0.47021778512935025, "f1": 0.7464691698243197, "recall": 0.7641043723554302},
    "met_r": {"mcc": 0.4161661511145206, "f1": 0.5316315205327414, "recall": 0.4897750511247444},
    "phos_y": {"mcc": 0.34001586910823867, "f1": 0.41832429174201324, "recall": 0.3395303326810176},
    "sumo_k": {"mcc": 0.47957386423314613, "f1": 0.7168961201501878, "recall": 0.7761517615176152},
}


# Provided by user (baseline MLP)
# Note: met_r was not provided for MLP.
BASELINE_MLP_RESULTS = {
    "acet_k": {"mcc": 0.37824148572270755, "f1": 0.702600309970725, "recall": 0.7193229901269393},
    "phos_y": {"mcc": 0.32538284526017164, "f1": 0.4601769911504425, "recall": 0.6360078277886497},
    "sumo_k": {"mcc": 0.3723462265227302, "f1": 0.6566747110377239, "recall": 0.7005420054200542},
}


# Provided by user (CNN+GRU classifier)
CNN_GRU_RESULTS = {
    "acet_k": {"mcc": 0.501944, "f1": 0.738904, "recall": 0.773514, "accuracy": 0.751553},
    "met_r": {"mcc": 0.487541, "f1": 0.576192, "recall": 0.621517, "accuracy": 0.857526},
    "phos_y": {"mcc": 0.444938, "f1": 0.534024, "recall": 0.627967, "accuracy": 0.846861},
    "sumo_k": {"mcc": 0.471753, "f1": 0.736490, "recall": 1.000000, "accuracy": 0.735678},
}


# Provided by user (MultiClassifier model)
# Best run (by macro MCC): 20260327_225704
# Per-PTM values provided: MCC, F1, AUROC, AUPR, N
# Note: recall/accuracy are not provided for this run.
MULTICLASSIFIER_RESULTS = {
    "acet_k": {"mcc": 0.4560, "f1": 0.7474},
    "met_r": {"mcc": 0.3843, "f1": 0.5146},
    "phos_y": {"mcc": 0.3661, "f1": 0.4744},
    "sumo_k": {"mcc": 0.4588, "f1": 0.7106},
}


def _as_float_or_none(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    try:
        value = str(value).strip()
        if value == "" or value.lower() in {"nan", "none", "null"}:
            return None
        return float(value)
    except Exception:
        return None


def results_to_frame(model_results: dict[str, dict[str, dict]]) -> pd.DataFrame:
    rows: list[dict] = []
    ptms: set[str] = set()
    for _model, per_ptm in model_results.items():
        ptms.update(per_ptm.keys())

    for ptm in sorted(ptms):
        if ptm in EXCLUDED_PTMS:
            continue
        for model, per_ptm in model_results.items():
            metrics = per_ptm.get(ptm, {})
            rows.append(
                {
                    "ptm": ptm,
                    "model": model,
                    "mcc": _as_float_or_none(metrics.get("mcc")),
                    "f1": _as_float_or_none(metrics.get("f1")),
                    "recall": _as_float_or_none(metrics.get("recall")),
                    "accuracy": _as_float_or_none(metrics.get("accuracy")),
                }
            )

    df = pd.DataFrame(rows)
    df["ptm"] = df["ptm"].astype(str)
    df["model"] = df["model"].astype(str)
    return df


def plot_grouped_bars(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    plot_df = df[["ptm", "model", metric]].copy()

    ptms = sorted(plot_df["ptm"].unique().tolist())
    models = [
        m
        for m in [
            "ptmgpt2 (heldout)",
            "baseline_mlp",
            "CNN+GRU",
            "MultiClassifier",
        ]
        if m in plot_df["model"].unique().tolist()
    ]
    # Append any other models (if script is extended later).
    for m in sorted(set(plot_df["model"].unique().tolist()) - set(models)):
        models.append(m)

    x = np.arange(len(ptms))
    n_models = max(len(models), 1)
    bar_width = min(0.8 / n_models, 0.22)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_width

    fig_w = max(8.0, 1.2 * len(ptms))
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))

    for idx, model in enumerate(models):
        sub = plot_df[plot_df["model"] == model].set_index("ptm")
        ys = [sub.at[ptm, metric] if ptm in sub.index else np.nan for ptm in ptms]
        ys = np.array([np.nan if v is None else v for v in ys], dtype=float)
        mask = ~np.isnan(ys)
        if mask.any():
            ax.bar(
                x[mask] + offsets[idx],
                ys[mask],
                width=bar_width,
                label=model,
                color=MODEL_COLORS.get(model, NORDIC_BASE_RGB),
                edgecolor=NORDIC_EDGE_RGB,
                linewidth=0.6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ptms, rotation=0)
    ax.set_ylabel(metric)
    ax.set_title(f"Model comparison: {metric}")
    if metric == "mcc":
        ax.set_ylim(-1.0, 1.0)
    else:
        ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.65)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "docs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_results = {
        "ptmgpt2 (heldout)": PTMGPT2_HELDOUT_RESULTS,
        "baseline_mlp": BASELINE_MLP_RESULTS,
        "CNN+GRU": CNN_GRU_RESULTS,
        "MultiClassifier": MULTICLASSIFIER_RESULTS,
    }

    df = results_to_frame(model_results)

    # Save merged table for easy reuse.
    df.sort_values(["ptm", "model"], inplace=True)
    df.to_csv(out_dir / "model_comparison_metrics.csv", index=False)

    for metric in METRICS:
        plot_grouped_bars(df, metric, out_dir / f"comparison_{metric}.png")

    # Lightweight console summary
    missing_accuracy_models = sorted(df[df["accuracy"].isna()]["model"].unique().tolist())
    if missing_accuracy_models:
        print(
            "Note: accuracy is missing for models: "
            + ", ".join(missing_accuracy_models)
            + ". (ptmgpt2 + MLP accuracy is not present in the provided files.)"
        )

    print(f"Wrote: {out_dir / 'model_comparison_metrics.csv'}")
    for metric in METRICS:
        print(f"Wrote: {out_dir / f'comparison_{metric}.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
