#!/usr/bin/env python3
"""
Regenerate fig3 calibration scatter for single-column IEEE layout.
Top panel:    LGBM (a only)
Bottom panel: LGBM Joint (a + c)
Output: fig3_calibration.pdf
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PRED = HERE / "analysis" / "predictions_all.csv"
OUT  = HERE / "figures" / "fig3_calibration.pdf"

df = pd.read_csv(PRED, low_memory=False)
df = df.dropna(subset=["pred_lgbm_sizey_MB", "pred_lgbm_joint_MB", "M_MB"])
df = df[df["M_MB"] > 0].copy()
print(f"Rows before per-workflow cap: {len(df):,}")

# Cap each workflow at MAX_PER_WF rows so dominant workflows
# don't drown out the tail. Sampling is reproducible (seed=42).
MAX_PER_WF = 1500
parts = []
rng = np.random.RandomState(42)
for wf, grp in df.groupby("workflow", sort=False):
    if len(grp) > MAX_PER_WF:
        idx = rng.choice(grp.index.values, size=MAX_PER_WF, replace=False)
        parts.append(grp.loc[idx])
    else:
        parts.append(grp)
df = pd.concat(parts, ignore_index=True)
print(f"Rows after per-workflow cap (max {MAX_PER_WF}): {len(df):,}")
print(df.groupby("workflow").size().rename("n").to_string())

# Paper-grade typography for single-column figure
plt.rcParams.update({
    "font.size":         18,
    "axes.titlesize":    22,
    "axes.labelsize":    20,
    "xtick.labelsize":   17,
    "ytick.labelsize":   17,
    "legend.fontsize":   17,
    "lines.linewidth":   3.0,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "axes.linewidth":    1.4,
    "figure.constrained_layout.use": True,
})

COL_SIZEY = "#7f7f7f"  # gray
COL_JOINT = "#16a085"  # teal

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=False, sharey=False)

panels = [
    ("pred_lgbm_sizey_MB", "LGBM (a only)",      COL_SIZEY),
    ("pred_lgbm_joint_MB", "LGBM Joint (a + c)", COL_JOINT),
]

for ax, (col, title, color) in zip(axes, panels):
    P = df[col].values
    M = df["M_MB"].values
    mask = np.isfinite(P) & (P > 0)
    P, M = P[mask], M[mask]

    ax.scatter(M, P, alpha=0.40, s=80, color=color,
                edgecolor="white", linewidth=0.6, zorder=2)

    lo = max(min(M.min(), P.min()) * 0.7, 1e-1)
    hi = max(M.max(), P.max()) * 1.4
    xs = np.array([lo, hi])
    ax.plot(xs, xs,       "k--", linewidth=2.8, label="perfect",  zorder=3)
    ax.plot(xs, 2 * xs,   "r:",  linewidth=2.4, alpha=0.85, label=r"$2\times$ over",  zorder=3)
    ax.plot(xs, 0.5 * xs, "r:",  linewidth=2.4, alpha=0.85, label=r"$2\times$ under", zorder=3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual peak RSS (MB)")
    ax.set_ylabel("Predicted M (MB)")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95,
                edgecolor="black", borderpad=0.5)
    ax.grid(True, which="both", alpha=0.3)

fig.savefig(OUT, format="pdf", bbox_inches="tight")
print(f"Wrote {OUT}")

# Also save a PNG companion at the same DPI for previewing
png = OUT.with_suffix(".png")
fig.savefig(png, dpi=150, bbox_inches="tight")
print(f"Wrote {png}")
plt.close(fig)
