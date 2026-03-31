import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

# -----------------------------
# Data
# -----------------------------
metrics = ["MCC", "Accuracy", "Precision", "Recall", "Fr-Score"]

baseline_mlp = [0.32, 0.66, 0.38, 0.58, 0.46]
cnn_gru      = [0.52, 0.72, 0.72, 0.678, 0.70]
cnn_bilstm   = [0.49,  0.71, 0.70, 0.68, 0.61]

# PTM-GPT2 results are pulled from ptmgpt2_best_checkpoints_summary.csv.
# Accuracy is not available in that file, so it is plotted as NaN.
PTMGPT2_PTM = "acet_k"
PTMGPT2_SPLIT = "heldout"


def load_ptmgpt2_series(ptm: str, data_type: str):
    csv_path = Path(__file__).resolve().parents[1] / "ptmgpt2_best_checkpoints_summary.csv"
    if not csv_path.exists():
        return None

    import csv

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ptm") == ptm and row.get("data_type") == data_type:
                mcc = float(row["mcc"])
                precision = float(row["precision"])
                recall = float(row["recall"])
                f1 = float(row["f1"])
                return [mcc, np.nan, precision, recall, f1]
    return None


ptmgpt2 = load_ptmgpt2_series(PTMGPT2_PTM, PTMGPT2_SPLIT)
if ptmgpt2 is None:
    ptmgpt2 = [np.nan] * len(metrics)



# -----------------------------
# Colors sampled to match chart
# -----------------------------
BG_COLOR       = "#ffffff"   # white background
TEXT_COLOR     = "#1f2836"   # dark navy labels/title
BASELINE_COLOR = "#6b7280"   # dark gray
GRU_COLOR      = "#8b0000"   # dark red
BILSTM_COLOR   = "#9ca3af"   # light gray
GPT2_COLOR     = "#2563eb"   # blue
GRID_COLOR     = "#d1d5db"   # light gray dashed grid

# -----------------------------
# Font settings
# Try Georgia first; fallback to serif
# -----------------------------
plt.rcParams["font.family"] = ["Georgia", "Times New Roman", "DejaVu Serif", "serif"]

# -----------------------------
# Figure
# -----------------------------
fig, ax = plt.subplots(figsize=(16.92, 3.52), dpi=100)
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

x = np.arange(len(metrics))
w = 0.20
bar_width = w * 0.75

bars1 = ax.bar(x - 1.5 * w, baseline_mlp, width=bar_width, color=BASELINE_COLOR, label="Baseline MLP")
bars2 = ax.bar(x - 0.5 * w, cnn_gru,      width=bar_width, color=GRU_COLOR,      label="CNN+GRU (Optimal)")
bars3 = ax.bar(x + 0.5 * w, cnn_bilstm,   width=bar_width, color=BILSTM_COLOR,   label="CNN+BiLSTM")
bars4 = ax.bar(x + 1.5 * w, ptmgpt2,      width=bar_width, color=GPT2_COLOR,     label=f"PTM-GPT2 ({PTMGPT2_PTM} {PTMGPT2_SPLIT})")

# -----------------------------
# Axes / grid / ticks
# -----------------------------
ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.01, 0.2))

# Match labels like 0, 0.2, 0.4 ... 1
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda y, _: f"{int(y)}" if abs(y - round(y)) < 1e-9 else f"{y:.1f}".rstrip("0").rstrip("."))
)

ax.grid(axis="y", linestyle="--", linewidth=1.2, color=GRID_COLOR, alpha=0.95)
ax.set_axisbelow(True)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=18, color=TEXT_COLOR)

ax.tick_params(axis="y", labelsize=14, colors=TEXT_COLOR)
ax.tick_params(axis="x", length=6, width=1.2, colors=TEXT_COLOR)

# Spines
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(TEXT_COLOR)
ax.spines["bottom"].set_linewidth(1.2)

# -----------------------------
# "Score" label at top-left
# -----------------------------
ax.text(
    -0.58, 1.03, "Score",
    transform=ax.transData,
    color=TEXT_COLOR,
    fontsize=19,
    ha="left",
    va="bottom"
)

# -----------------------------
# Legend
# -----------------------------
legend_handles = [
    Patch(facecolor=BASELINE_COLOR, edgecolor=BASELINE_COLOR, label="Baseline MLP"),
    Patch(facecolor=GRU_COLOR,      edgecolor=GRU_COLOR,      label="CNN+GRU (Optimal)"),
    Patch(facecolor=BILSTM_COLOR,   edgecolor=BILSTM_COLOR,   label="CNN+BiLSTM"),
    Patch(facecolor=GPT2_COLOR,     edgecolor=GPT2_COLOR,     label=f"PTM-GPT2 ({PTMGPT2_PTM} {PTMGPT2_SPLIT})"),
]

leg = ax.legend(
    handles=legend_handles,
    ncol=4,
    loc="upper center",
    bbox_to_anchor=(0.52, 1.18),
    frameon=False,
    fontsize=19,
    handlelength=1.3,
    handletextpad=0.45,
    columnspacing=0.9,
)
for t in leg.get_texts():
    t.set_color(TEXT_COLOR)

# Rounded-looking legend swatches (Matplotlib API differs by version)
_legend_handles = getattr(leg, "legend_handles", None)
if _legend_handles is None:
    _legend_handles = getattr(leg, "legendHandles", None)
if _legend_handles is None:
    _legend_handles = []

for h in _legend_handles:
    try:
        h.set_linewidth(0)
    except Exception:
        pass
    try:
        h.set_joinstyle("round")
    except Exception:
        pass

# -----------------------------
# Value labels on bars
# -----------------------------
def fmt(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")

def add_labels(bars):
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.text(
            b.get_x() + b.get_width()/2,
            h + 0.015,
            fmt(h),
            ha="center",
            va="bottom",
            fontsize=15,
            color=TEXT_COLOR,
            fontweight="bold"
        )

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

plt.tight_layout(pad=0.8)
plt.show()

# Optional save:
# plt.savefig("model_comparison_chart.png", dpi=300, facecolor=BG_COLOR, bbox_inches="tight")