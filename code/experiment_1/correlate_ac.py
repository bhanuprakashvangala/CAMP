"""Per-bucket correlation between a (file size) and c (bytes consumed)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sys
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

REPO = Path("/shared/training")
EPS = 1.0
plt.rcParams.update({"font.size":18,"axes.titlesize":22,"axes.labelsize":20,
    "xtick.labelsize":16,"ytick.labelsize":16,"legend.fontsize":16,
    "lines.markersize":12,"lines.linewidth":3.0,"axes.linewidth":1.8,
    "savefig.dpi":300,"savefig.bbox":"tight"})

ours = pd.read_csv(REPO/"all_workflows.csv")
on = pd.DataFrame({"workflow":ours["workflow"], "process":ours["process"],
    "a":pd.to_numeric(ours["a_bytes"],errors="coerce"),
    "c":pd.to_numeric(ours["c_bytes"],errors="coerce"),
    "M":pd.to_numeric(ours["M_peak_rss_bytes"],errors="coerce"),
}).dropna(subset=["a","c","M"]).query("M>0 and c>0 and a>=0").reset_index(drop=True)
on = on[on["workflow"] != "methylseq"]
sys.path.insert(0, str(REPO))
from load_pyradiomics import load_pyradiomics
pn = load_pyradiomics(str(REPO/"pyradiomics_32k.csv"))[["workflow","process","a","c","M"]]
df = pd.concat([on, pn], ignore_index=True)
print(f"c-present rows: {len(df):,}")

# ===== Per-bucket correlation =====
print()
print(f"{'bucket':<55}{'n':>6}{'pearson(log a, log c)':>24}{'spearman':>11}{'R² LR(a->c)':>13}{'c/a ratio':>13}")
print("-"*125)

results = []
for (wf, proc), grp in df.groupby(["workflow","process"]):
    if len(grp) < 10: continue
    a, c = grp["a"].values, grp["c"].values
    la, lc = np.log(a + EPS), np.log(c + EPS)
    pe, _ = pearsonr(la, lc)
    sp, _ = spearmanr(la, lc)
    lr = LinearRegression().fit(la.reshape(-1,1), lc)
    r2 = lr.score(la.reshape(-1,1), lc)
    ratio_med = float(np.median(c / np.maximum(a, 1)))
    results.append({"bucket":f"{wf}::{proc}", "n":len(grp), "pearson":pe, "spearman":sp, "r2":r2, "c_over_a":ratio_med})
    print(f"{wf+'::'+proc[:50]:<55}{len(grp):>6}{pe:>24.3f}{sp:>11.3f}{r2:>13.3f}{ratio_med:>13.2f}x")

print()
print("="*125)
res_df = pd.DataFrame(results)
print(f"AGGREGATE across {len(res_df)} buckets:")
print(f"  Pearson(log a, log c):  median={res_df.pearson.median():.3f}  mean={res_df.pearson.mean():.3f}  min={res_df.pearson.min():.3f}  max={res_df.pearson.max():.3f}")
print(f"  Spearman rank corr:     median={res_df.spearman.median():.3f}  mean={res_df.spearman.mean():.3f}")
print(f"  R² of LR(log a -> log c):  median={res_df.r2.median():.3f}  mean={res_df.r2.mean():.3f}")
print(f"  Median c/a ratio (raw bytes): median across buckets = {res_df.c_over_a.median():.2f}x")

# ===== Figure 1: scatter (a, c) per bucket — small multiples =====
fig, axes = plt.subplots(3, 5, figsize=(22, 13))
axes = axes.flatten()
buckets_to_plot = list(df.groupby(["workflow","process"]))
buckets_to_plot = [b for b in buckets_to_plot if len(b[1]) >= 10][:14]
for i, ((wf, proc), grp) in enumerate(buckets_to_plot):
    ax = axes[i]
    a, c = grp["a"].values, grp["c"].values
    ax.scatter(a/1024**2, c/1024**2, alpha=0.4, s=80, edgecolor="white", linewidth=0.5)
    lo = min(a.min(), c.min())/1024**2; hi = max(a.max(), c.max())/1024**2
    ax.plot([max(lo,0.001), hi], [max(lo,0.001), hi], "k--", linewidth=2, alpha=0.7, label="c=a")
    pe, _ = pearsonr(np.log(a+EPS), np.log(c+EPS))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{proc[:36]}\nn={len(grp)} pearson={pe:.2f}", fontsize=14)
    ax.set_xlabel("a (MB)", fontsize=14); ax.set_ylabel("c (MB)", fontsize=14)
    ax.tick_params(labelsize=12); ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=12, loc="lower right")
for j in range(i+1, len(axes)): axes[j].axis("off")
fig.suptitle("Per-bucket scatter:  c (bytes consumed) vs a (file size)", fontsize=24)
fig.tight_layout()
fig.savefig(REPO/"figures/fig7_a_vs_c_scatter.png"); plt.close()
print(f"\nwrote fig7_a_vs_c_scatter.png")

# ===== Figure 2: distribution of correlations =====
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].hist(res_df.pearson, bins=15, color="#16a085", edgecolor="black", linewidth=2)
axes[0].axvline(res_df.pearson.median(), color="red", linewidth=3, linestyle="--",
                label=f"median = {res_df.pearson.median():.3f}")
axes[0].set_xlabel("Pearson correlation (log a, log c)")
axes[0].set_ylabel("Number of buckets")
axes[0].set_title("Distribution of per-bucket Pearson correlations")
axes[0].legend(); axes[0].grid(axis="y", alpha=0.4, linestyle="--")

axes[1].hist(res_df.r2, bins=15, color="#e74c3c", edgecolor="black", linewidth=2)
axes[1].axvline(res_df.r2.median(), color="navy", linewidth=3, linestyle="--",
                label=f"median = {res_df.r2.median():.3f}")
axes[1].set_xlabel("R² of LR(log a → log c)")
axes[1].set_ylabel("Number of buckets")
axes[1].set_title("How well does a alone predict c?")
axes[1].legend(); axes[1].grid(axis="y", alpha=0.4, linestyle="--")
fig.tight_layout()
fig.savefig(REPO/"figures/fig8_correlation_dist.png"); plt.close()
print(f"wrote fig8_correlation_dist.png")
