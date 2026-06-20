#!/usr/bin/env python3
"""Replot the AL curves with the same titling as the paper figure.

Reads results_active_learning.csv and writes both
  reruns/figures/fig_e3_al_curves.{png,pdf}
and copies them to
  IEEE_Cluster_paper/files/Figures/fig_e3_al_curves.{png,pdf}
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
CSV  = REPO / "results_active_learning.csv"
OUT_DIR = REPO / "figures"
OUT_DIR.mkdir(exist_ok=True)
PAPER_FIG_DIR = REPO.parent.parent / "IEEE_Cluster_paper" / "files" / "Figures"

out = pd.read_csv(CSV)

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16,
    "legend.fontsize": 14, "lines.linewidth": 3.0, "lines.markersize": 12,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.constrained_layout.use": True,
})
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
color  = {"gate": "#d62728", "random": "#1f77b4", "oracle": "#2ca02c"}
marker = {"gate": "^",        "random": "o",        "oracle": "*"}

for col, exp in enumerate(["exp1", "exp2"]):
    d = out[(out["experiment"] == exp) & (out["strategy"].isin(["gate", "random"]))]
    oracle = out[(out["experiment"] == exp) & (out["strategy"] == "oracle")]
    oracle_row = oracle.iloc[0] if len(oracle) else None
    title_p = "Exp 1 (point)" if exp == "exp1" else "Exp 2 (distribution)"

    for metric, ylabel, ax in [
        ("MAPE_med",   "Median APE (%)",     axes[0, col]),
        ("Wastage_MB", "Wastage (MB-units)", axes[1, col]),
    ]:
        for strat in ["random", "gate"]:
            sub = d[d["strategy"] == strat].sort_values("round")
            ax.plot(sub["cumul_audit_frac"], sub[metric],
                    color=color[strat], marker=marker[strat],
                    label=strat.capitalize() + " AL")
        if oracle_row is not None:
            ax.axhline(oracle_row[metric], color=color["oracle"], linestyle="--",
                       linewidth=2.5, label="Full audit (B=1)")
        ax.set_xlabel("Cumulative audit fraction")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_p} — {ylabel}")
        ax.set_xlim(-0.02, 0.32)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Experiment 3 — Active-learning convergence vs random sampling",
             fontsize=20, y=1.02)

png = OUT_DIR / "fig_e3_al_curves.png"
pdf = OUT_DIR / "fig_e3_al_curves.pdf"
fig.savefig(png, dpi=150, bbox_inches="tight")
fig.savefig(pdf, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"wrote {png}")
print(f"wrote {pdf}")

if PAPER_FIG_DIR.exists():
    for src in (png, pdf):
        dst = PAPER_FIG_DIR / src.name
        shutil.copyfile(src, dst)
        print(f"copied -> {dst}")
else:
    print(f"paper dir not found, skipped copy: {PAPER_FIG_DIR}")
