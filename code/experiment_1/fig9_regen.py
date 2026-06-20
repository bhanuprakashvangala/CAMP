#!/usr/bin/env python3
"""
Regenerate fig9_per_model_calibration.{png,pdf} from analysis/predictions_all.csv.

Big bold typography for paper print:
  - title 30 bold, suptitle 32 bold
  - per-cell title 26 bold (model name)
  - row label (Sizey / Joint) 26 bold
  - axis labels 24 bold, ticks 20 bold
  - MAPE annotation 22 bold
"""
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PRED = HERE / "analysis" / "predictions_all.csv"
OUT_PNG = HERE / "figures" / "fig9_per_model_calibration.png"
OUT_PDF = HERE / "figures" / "fig9_per_model_calibration.pdf"

# Per-workflow row cap so iwd / pyradiomics don't drown the others.
MAX_PER_WF = 1500

df = pd.read_csv(PRED, low_memory=False)
df = df.dropna(subset=["M_MB"]).query("M_MB > 0").copy()
# The paper's per_task_paper.pkl was only fit on iwd + pyradiomics buckets.
# Predictions for the other workflows in this CSV come from fall-through
# fallbacks and are not meaningful — filtering them out brings MAPE back
# to the per-bucket-fit regime.
df = df[df["workflow"].isin(["iwd", "pyradiomics"])].reset_index(drop=True)

rng = np.random.RandomState(42)
parts = []
for wf, grp in df.groupby("workflow", sort=False):
    if len(grp) > MAX_PER_WF:
        idx = rng.choice(grp.index.values, size=MAX_PER_WF, replace=False)
        parts.append(grp.loc[idx])
    else:
        parts.append(grp)
df = pd.concat(parts, ignore_index=True)
print(f"plotting {len(df):,} rows after per-workflow cap of {MAX_PER_WF}")

plt.rcParams.update({
    "font.size":          22,
    "axes.titlesize":     26,
    "axes.titleweight":   "bold",
    "axes.labelsize":     24,
    "axes.labelweight":   "bold",
    "xtick.labelsize":    20,
    "ytick.labelsize":    20,
    "legend.fontsize":    20,
    "figure.titlesize":   32,
    "figure.titleweight": "bold",
    "lines.markersize":   12,
    "lines.linewidth":    3.0,
    "axes.linewidth":     2.0,
    "xtick.major.width":  2.0,
    "ytick.major.width":  2.0,
    "xtick.major.size":   7,
    "ytick.major.size":   7,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "font.family":        "DejaVu Sans",
})

MODELS = [
    ("lr",   "Linear Regression"),
    ("knn",  "KNN (k=5)"),
    ("mlp",  "MLP (64-32)"),
    ("rf",   "Random Forest"),
    ("lgbm", "LightGBM"),
    ("ngb",  "NGBoost LogNormal"),
]

C_SIZEY = "#6c757d"   # gray
C_JOINT = "#16a085"   # teal

fig, axes = plt.subplots(2, 6, figsize=(30, 12), sharex=True, sharey=True)

for col, (key, label) in enumerate(MODELS):
    for row, variant in enumerate(["sizey", "joint"]):
        ax = axes[row, col]
        pcol = f"pred_{key}_{variant}_MB"
        if pcol not in df.columns:
            ax.text(0.5, 0.5, "(no fit)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=22, fontweight="bold")
            ax.set_xscale("log"); ax.set_yscale("log")
            continue
        sub = df[["M_MB", pcol]].dropna()
        sub = sub[(sub["M_MB"] > 0) & (sub[pcol] > 0)
                  & np.isfinite(sub[pcol]) & np.isfinite(sub["M_MB"])]
        # filter pathological predictions (numerical blow-ups from
        # exponentiated log-space outputs) — keep within 10000x of M
        sub = sub[(sub[pcol] / sub["M_MB"] >= 1e-4)
                  & (sub[pcol] / sub["M_MB"] <= 1e4)]
        if len(sub) == 0:
            ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=22, fontweight="bold")
            ax.set_xscale("log"); ax.set_yscale("log")
            continue

        M = sub["M_MB"].values
        P = sub[pcol].values
        mape = float(np.mean(np.abs(P - M) / M) * 100)
        color = C_SIZEY if variant == "sizey" else C_JOINT

        ax.scatter(M, P, alpha=0.40, s=55, color=color,
                   edgecolor="white", linewidth=0.5, zorder=2)
        # axis limits are driven by ACTUAL M only — keeps the panel readable
        # when a single model's prediction goes haywire
        lo = max(M.min() * 0.5, 1e-1)
        hi = M.max() * 2.0
        xs = np.array([lo, hi])
        ax.plot(xs, xs, "k--", linewidth=2.8, label="perfect", zorder=3)
        ax.plot(xs, 2 * xs, "r:", linewidth=2.2, alpha=0.85, zorder=3)
        ax.plot(xs, 0.5 * xs, "r:", linewidth=2.2, alpha=0.85, zorder=3)

        ax.text(0.05, 0.92, f"MAPE = {mape:.2f}%",
                transform=ax.transAxes,
                fontsize=22, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="black", linewidth=1.6))

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.grid(True, which="both", alpha=0.30)

        if row == 0:
            ax.set_title(label)
        if col == 0:
            ax.set_ylabel(("Sizey  (a only)" if variant == "sizey"
                           else "Joint  (a, c)"))
        if row == 1:
            ax.set_xlabel("Actual M (MB)")

# global y-label (overrides individual)
fig.text(0.06, 0.5, "Predicted M (MB)", rotation=90,
         fontsize=28, fontweight="bold", va="center")

fig.suptitle("Per-model calibration  —  predicted vs actual peak RSS  (held-out test)",
             y=0.995)
fig.tight_layout(rect=[0.075, 0, 1, 0.96])

fig.savefig(OUT_PNG)
fig.savefig(OUT_PDF, format="pdf")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
plt.close(fig)
