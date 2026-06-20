#!/usr/bin/env python3
"""
Paper-grade summary figures for Experiment 1.

Layout: paired bars per model class (a only vs Joint a+c),
log-scale y-axes for wastage and OOM count, clean palette.

Outputs (PDF + PNG):
  fig_a_aggregated_wastage.{pdf,png}   - per-model paired wastage
  fig_b_oom_counts.{pdf,png}           - per-model paired OOM count
  fig_c_wastage_per_workflow.{pdf,png} - per-workflow heatmap
  fig_d_runtime_per_workflow.{pdf,png} - per-workflow runtime
  table_method_summary.csv             - master numbers
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

HERE = Path(__file__).resolve().parent
PRED = HERE / "analysis" / "predictions_all.csv"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

df = pd.read_csv(PRED, low_memory=False)
df = df.dropna(subset=["M_MB", "runtime_seconds"])
df = df[(df["M_MB"] > 0) & (df["runtime_seconds"] > 0)].copy()
print(f"Rows: {len(df):,}; workflows: {df['workflow'].nunique()}")

# Best-of-4 ensemble emulation
sizey4 = ["lr", "knn", "mlp", "rf"]
err = pd.concat(
    [(df[f"safe_{m}_sizey_MB"] - df["M_MB"]).abs().rename(m) for m in sizey4],
    axis=1)
df["_pick"] = err.idxmin(axis=1)
df["safe_best4_MB"] = df.apply(lambda r: r[f"safe_{r['_pick']}_sizey_MB"], axis=1)

CLASSES = ["LR", "KNN", "MLP", "RF", "LGBM", "NGB"]
SIZEY_COL = {"LR": "safe_lr_sizey_MB", "KNN": "safe_knn_sizey_MB",
              "MLP": "safe_mlp_sizey_MB", "RF": "safe_rf_sizey_MB",
              "LGBM": "safe_lgbm_sizey_MB", "NGB": "safe_ngb_sizey_MB"}
JOINT_COL = {"LR": "safe_lr_joint_MB", "KNN": "safe_knn_joint_MB",
              "MLP": "safe_mlp_joint_MB", "RF": "safe_rf_joint_MB",
              "LGBM": "safe_lgbm_joint_MB", "NGB": "safe_ngb_joint_MB"}

def aggr_wastage(col):
    sub = df.dropna(subset=[col]).copy()
    over = np.maximum(sub[col] - sub["M_MB"], 0.0)
    return float((over * sub["runtime_seconds"] / 3600.0 / 1024.0).sum())

def aggr_oom(col):
    sub = df.dropna(subset=[col])
    return int((sub[col] < sub["M_MB"]).sum())

w_sizey = [aggr_wastage(SIZEY_COL[c]) for c in CLASSES]
w_joint = [aggr_wastage(JOINT_COL[c]) for c in CLASSES]
w_best4 = aggr_wastage("safe_best4_MB")

o_sizey = [aggr_oom(SIZEY_COL[c]) for c in CLASSES]
o_joint = [aggr_oom(JOINT_COL[c]) for c in CLASSES]
o_best4 = aggr_oom("safe_best4_MB")

# Save the master CSV for the LaTeX table
def aggr_per_wf(col, fn):
    sub = df.dropna(subset=[col]).copy()
    if fn == "wastage":
        sub["_v"] = np.maximum(sub[col] - sub["M_MB"], 0.0) * sub["runtime_seconds"] / 3600.0 / 1024.0
    else:
        sub["_v"] = (sub[col] < sub["M_MB"]).astype(int)
    return sub.groupby("workflow")["_v"].sum()

rows = []
for cls in CLASSES:
    for var, col in [("a only", SIZEY_COL[cls]), ("Joint (a+c)", JOINT_COL[cls])]:
        rows.append({"class": cls, "variant": var,
                       "wastage_GBh_total": aggr_wastage(col),
                       "oom_total":         aggr_oom(col)})
rows.append({"class": "Best-of-4", "variant": "a only",
              "wastage_GBh_total": w_best4, "oom_total": o_best4})
pd.DataFrame(rows).to_csv(FIGS / "table_method_summary.csv", index=False)
print("Wrote", FIGS / "table_method_summary.csv")

# === styling ============================================================
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        18,
    "axes.titlesize":   23,
    "axes.titleweight": "bold",
    "axes.labelsize":   20,
    "axes.labelweight": "bold",
    "xtick.labelsize":  18,
    "ytick.labelsize":  16,
    "legend.fontsize":  17,
    "axes.linewidth":   1.6,
    "axes.edgecolor":   "#333333",
    "axes.grid":        True,
    "axes.axisbelow":   True,
    "grid.color":       "#cccccc",
    "grid.alpha":       0.7,
    "grid.linestyle":   "-",
    "grid.linewidth":   0.7,
    "figure.constrained_layout.use": True,
    "savefig.facecolor": "white",
})

C_SIZEY  = "#b8b8b8"   # light gray
C_JOINT  = "#0e7a6f"   # deep teal
C_BEST4  = "#d97706"   # warm amber

# === Figure A: paired wastage bars =======================================
fig, ax = plt.subplots(figsize=(13, 6.5))
n = len(CLASSES)
x = np.arange(n)
w = 0.36
b1 = ax.bar(x - w/2, w_sizey, w, color=C_SIZEY, edgecolor="black",
              linewidth=1.4, label="a only", zorder=3)
b2 = ax.bar(x + w/2, w_joint, w, color=C_JOINT, edgecolor="black",
              linewidth=1.4, label="Joint (a+c)", zorder=3)
# Best-of-4 sits alone on the right
b3 = ax.bar([n + 0.6], [w_best4], 0.55, color=C_BEST4, edgecolor="black",
              linewidth=1.4, label="Best-of-4 (a, baseline)", zorder=3)

ax.set_yscale("log")
ax.set_ylim(1.0, 200)
ax.set_xticks(list(x) + [n + 0.6])
ax.set_xticklabels(CLASSES + ["Best-of-4"])
ax.set_ylabel("Wastage (GB-hours, log)", labelpad=6)
ax.set_title("Aggregated memory wastage across all workflows", pad=14)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
ax.legend(loc="upper right", frameon=True, framealpha=0.97,
            edgecolor="#333333", fancybox=False, borderpad=0.6)

def annotate(bars, vals, fmt="{:.2f}"):
    for bar, v in zip(bars, vals):
        if v <= 0: continue
        ax.annotate(fmt.format(v),
                      xy=(bar.get_x() + bar.get_width()/2, v),
                      xytext=(0, 5), textcoords="offset points",
                      ha="center", va="bottom",
                      fontsize=14, fontweight="bold")
annotate(b1, w_sizey)
annotate(b2, w_joint)
annotate(b3, [w_best4])

fig.savefig(FIGS / "fig_a_aggregated_wastage.pdf", bbox_inches="tight")
fig.savefig(FIGS / "fig_a_aggregated_wastage.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Wrote fig_a_aggregated_wastage.{pdf,png}")

# === Figure B: paired OOM bars ===========================================
fig, ax = plt.subplots(figsize=(13, 6.5))
b1 = ax.bar(x - w/2, o_sizey, w, color=C_SIZEY, edgecolor="black",
              linewidth=1.4, label="a only", zorder=3)
b2 = ax.bar(x + w/2, o_joint, w, color=C_JOINT, edgecolor="black",
              linewidth=1.4, label="Joint (a+c)", zorder=3)
b3 = ax.bar([n + 0.6], [o_best4], 0.55, color=C_BEST4, edgecolor="black",
              linewidth=1.4, label="Best-of-4 (a, baseline)", zorder=3)

ax.set_xticks(list(x) + [n + 0.6])
ax.set_xticklabels(CLASSES + ["Best-of-4"])
ax.set_ylabel("OOM-allocation events (count)", labelpad=6)
ax.set_title("Held-out OOMs by method (lower is better)", pad=14)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
ax.legend(loc="upper right", frameon=True, framealpha=0.97,
            edgecolor="#333333", fancybox=False, borderpad=0.6)

annotate(b1, o_sizey, fmt="{:.0f}")
annotate(b2, o_joint, fmt="{:.0f}")
annotate(b3, [o_best4], fmt="{:.0f}")

# Add some headroom for annotations
ax.set_ylim(0, max(max(o_sizey), max(o_joint), o_best4) * 1.15)

fig.savefig(FIGS / "fig_b_oom_counts.pdf", bbox_inches="tight")
fig.savefig(FIGS / "fig_b_oom_counts.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Wrote fig_b_oom_counts.{pdf,png}")

# === Figure C: per-workflow heatmap ======================================
wf_order = ["chipseq", "eager", "mag", "mag_karlsson", "methylseq",
              "methylseq_naga", "rnaseq", "iwd", "pyradiomics"]
mt_labels = []
mt_cols = []
for cls in CLASSES:
    mt_labels += [f"{cls} (a)", f"{cls} (a+c)"]
    mt_cols += [SIZEY_COL[cls], JOINT_COL[cls]]
mt_labels.append("Best-of-4 (a)"); mt_cols.append("safe_best4_MB")

W = pd.DataFrame(index=[w for w in wf_order if w in df["workflow"].unique()],
                   columns=mt_labels, dtype=float)
for lbl, col in zip(mt_labels, mt_cols):
    s = aggr_per_wf(col, "wastage")
    for wf in W.index:
        W.loc[wf, lbl] = float(s.get(wf, 0.0))

fig, ax = plt.subplots(figsize=(15, 7))
data = W.values.astype(float)
data_pos = np.where(data > 0, data, np.nan)
norm = mcolors.LogNorm(vmin=max(np.nanpercentile(data_pos, 2), 1e-3),
                          vmax=max(np.nanpercentile(data_pos, 99), 1.0))
im = ax.imshow(data_pos, cmap="YlOrRd", norm=norm, aspect="auto")
ax.set_xticks(range(len(W.columns)))
ax.set_xticklabels(W.columns, rotation=35, ha="right")
ax.set_yticks(range(len(W.index)))
ax.set_yticklabels(W.index)
ax.set_title("Per-workflow wastage (GB-hours)", pad=12)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        v = W.values[i, j]
        if pd.isna(v) or v == 0:
            txt = "0"
        elif v < 0.01:
            txt = f"{v:.3f}"
        else:
            txt = f"{v:.2f}"
        threshold = (norm.vmin * norm.vmax) ** 0.5
        ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                  color="black" if (pd.isna(v) or v < threshold) else "white",
                  fontweight="bold")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("GB-hours (log)", fontsize=15)
ax.grid(False)

fig.savefig(FIGS / "fig_c_wastage_per_workflow.pdf", bbox_inches="tight")
fig.savefig(FIGS / "fig_c_wastage_per_workflow.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Wrote fig_c_wastage_per_workflow.{pdf,png}")

# === Figure D: per-workflow runtime ======================================
runtime = (df.groupby("workflow")["runtime_seconds"].sum() / 3600.0)
runtime = runtime.reindex([w for w in wf_order if w in runtime.index])

fig, ax = plt.subplots(figsize=(11, 6))
xs = np.arange(len(runtime))
bars = ax.bar(xs, runtime.values, color="#1f6feb",
                edgecolor="black", linewidth=1.4, zorder=3)
ax.set_yscale("log")
ax.set_ylim(0.01, runtime.max() * 1.4)
ax.set_xticks(xs); ax.set_xticklabels(runtime.index, rotation=30, ha="right")
ax.set_ylabel("Total task runtime (hours, log)", labelpad=6)
ax.set_title("Aggregated task runtime per workflow", pad=14)
ax.grid(axis="x", visible=False)
ax.set_axisbelow(True)
for bar, v in zip(bars, runtime.values):
    label = f"{v:.0f} h" if v >= 1 else f"{v:.2f} h"
    ax.annotate(label,
                  xy=(bar.get_x() + bar.get_width()/2, v),
                  xytext=(0, 5), textcoords="offset points",
                  ha="center", va="bottom",
                  fontsize=15, fontweight="bold")

fig.savefig(FIGS / "fig_d_runtime_per_workflow.pdf", bbox_inches="tight")
fig.savefig(FIGS / "fig_d_runtime_per_workflow.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Wrote fig_d_runtime_per_workflow.{pdf,png}")
print("\nDone.")
