#!/usr/bin/env python3
"""Build the Experiment 3 operating-curve figure from results_exp3.csv."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
df = pd.read_csv(OUT / "results_exp3.csv").drop_duplicates(
    subset=["experiment", "B", "strategy", "retrain"])

# Use IPW where available, else naive (effectively the same here, but cleaner picks)
df["preferred"] = df["retrain"].map({"ipw": 0, "naive": 1, "n/a": 2})
df = df.sort_values(["experiment", "B", "strategy", "preferred"]).drop_duplicates(
    subset=["experiment", "B", "strategy"], keep="first")

plt.rcParams.update({
    "font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18,
    "legend.fontsize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16,
    "lines.linewidth": 3.0, "lines.markersize": 12,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.constrained_layout.use": True,
})

color = {"sizey": "#888888", "random": "#1f77b4", "gate": "#d62728", "oracle": "#2ca02c"}
marker = {"sizey": "s", "random": "o", "gate": "^", "oracle": "*"}
zorder = {"sizey": 1, "random": 2, "gate": 4, "oracle": 5}


def plot_panel(ax, dexp, metric, ylabel, title):
    # x-axis: B; lines per strategy
    for strat in ["random", "gate"]:
        sub = dexp[dexp["strategy"] == strat].sort_values("B")
        # For these strategies, also include B=0 (sizey) and B=1 (oracle) as anchor points
        sizey = dexp[dexp["strategy"] == "sizey"].iloc[0:1]
        oracle = dexp[dexp["strategy"] == "oracle"].iloc[0:1]
        xs = list(sizey["B"]) + list(sub["B"]) + list(oracle["B"])
        ys = list(sizey[metric]) + list(sub[metric]) + list(oracle[metric])
        ax.plot(xs, ys, color=color[strat], marker=marker[strat],
                  label=strat.capitalize() + " selection", zorder=zorder[strat])
    # Reference lines
    sizey = dexp[dexp["strategy"] == "sizey"].iloc[0]
    oracle = dexp[dexp["strategy"] == "oracle"].iloc[0]
    ax.axhline(sizey[metric], color=color["sizey"], linestyle=":", linewidth=2.5,
                  label=f"Sizey-only (B=0)", zorder=zorder["sizey"])
    ax.axhline(oracle[metric], color=color["oracle"], linestyle="--", linewidth=2.5,
                  label=f"Full audit (B=1)", zorder=zorder["oracle"])
    ax.set_xlabel("Audit budget $B$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)


fig, axes = plt.subplots(2, 2, figsize=(16, 11))

for col, exp in enumerate(["exp1", "exp2"]):
    dexp = df[df["experiment"] == exp].copy()
    title_prefix = "Exp 1 (point)" if exp == "exp1" else "Exp 2 (distribution)"
    plot_panel(axes[0, col], dexp, "MAPE_med",
                  "Median APE (%)", f"{title_prefix} — held-out MAPE")
    plot_panel(axes[1, col], dexp, "Wastage_MB",
                  "Wastage (MB-units)", f"{title_prefix} — held-out wastage")

# Single legend for the whole figure
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Experiment 3 — Active-learning gate operating curves",
              fontsize=22, y=1.02)

out_path = OUT / "figures" / "fig_e3_1_operating_curve.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Wrote {out_path}")
plt.close(fig)


# Second figure: bar comparison at B=0.1 and B=0.3, MAPE + R²_log + OOM
fig2, axes2 = plt.subplots(2, 3, figsize=(20, 11))
metrics = [("MAPE_med", "Median APE (%)"),
            ("R2_log", "$R^2_{\\log}$"),
            ("OOM_count", "OOM count")]

for col, (m, ylab) in enumerate(metrics):
    for row, exp in enumerate(["exp1", "exp2"]):
        ax = axes2[row, col]
        dexp = df[df["experiment"] == exp]
        # rows: sizey + random@0.1 + gate@0.1 + random@0.3 + gate@0.3 + oracle
        order = [
            ("Sizey",       "sizey",  0.0),
            ("Random@0.1",  "random", 0.1),
            ("Gate@0.1",    "gate",   0.1),
            ("Random@0.3",  "random", 0.3),
            ("Gate@0.3",    "gate",   0.3),
            ("Oracle",      "oracle", 1.0),
        ]
        labels, vals, colors = [], [], []
        for lbl, strat, b in order:
            r = dexp[(dexp["strategy"] == strat) & (dexp["B"] == b)]
            if len(r):
                labels.append(lbl); vals.append(float(r[m].iloc[0]))
                colors.append(color[strat])
        bars = ax.bar(range(len(labels)), vals, color=colors,
                        edgecolor="black", linewidth=1.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(f"{'Exp 1' if exp=='exp1' else 'Exp 2'} — {ylab}")
        for bar, v in zip(bars, vals):
            h = bar.get_height()
            ax.annotate(f"{v:.2f}" if v < 100 else f"{int(v)}",
                          xy=(bar.get_x() + bar.get_width()/2, h),
                          xytext=(0, 5), textcoords="offset points",
                          ha="center", va="bottom", fontsize=14)

fig2.suptitle("Experiment 3 — Strategy comparison at fixed budgets",
              fontsize=22, y=1.02)
out_path2 = OUT / "figures" / "fig_e3_2_strategy_bars.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Wrote {out_path2}")
plt.close(fig2)


# Third figure: gate component contribution (only Exp1 + Exp2 audit subsets at B=0.1)
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

for col, (exp_name, fp) in enumerate([
    ("Exp 1 (point)", "C:/Users/govin/Downloads/IEEECLUSTER/IEEE_CLUSTER_MAIN/selective_audit/experiment1/budge_0.1_Exp1/exp1_lgbm_sizey_scores_b_0.1.csv"),
    ("Exp 2 (distribution)", "C:/Users/govin/Downloads/IEEECLUSTER/IEEE_CLUSTER_MAIN/selective_audit/experiment1/experiment2/budget_0.1_Exp2/exp2_qlgbm_sizey_scores_b_0.1.csv"),
]):
    sc = pd.read_csv(fp)
    aud = sc[sc["audit_flag"] == "Audit"]
    no  = sc[sc["audit_flag"] == "NoAudit"]
    components = ["uncertainty_score", "risk_score", "novelty_score"]
    means_aud = [float(aud[c].mean()) for c in components]
    means_no  = [float(no[c].mean()) for c in components]
    x = np.arange(len(components))
    w = 0.38
    axes3[col].bar(x - w/2, means_aud, w, label="Audit", color="#d62728",
                      edgecolor="black", linewidth=1.5)
    axes3[col].bar(x + w/2, means_no, w, label="No audit", color="#888888",
                      edgecolor="black", linewidth=1.5)
    axes3[col].set_xticks(x)
    axes3[col].set_xticklabels(["G1 uncertainty", "G2 risk", "G3 novelty"])
    axes3[col].set_ylabel("Mean component score")
    axes3[col].set_title(f"{exp_name} @ B = 0.1")
    axes3[col].legend()

fig3.suptitle("Gate component drivers per experiment", fontsize=22, y=1.02)
out_path3 = OUT / "figures" / "fig_e3_3_gate_components.png"
fig3.savefig(out_path3, dpi=150, bbox_inches="tight")
print(f"Wrote {out_path3}")
plt.close(fig3)

print("\nDone.")
