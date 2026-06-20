"""Per-model calibration figures — one cell per model, both Sizey and Joint variants."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd, sys
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path("/shared/training")
EPS, MIN, SEED = 1.0, 10, 42

plt.rcParams.update({
    "font.size":          18,
    "axes.titlesize":     22,
    "axes.labelsize":     20,
    "xtick.labelsize":    16,
    "ytick.labelsize":    16,
    "legend.fontsize":    16,
    "figure.titlesize":   26,
    "lines.markersize":   12,
    "lines.linewidth":    3.0,
    "axes.linewidth":     1.8,
    "xtick.major.width":  1.5,
    "ytick.major.width":  1.5,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

with open(REPO/"models_unified/per_task_paper.pkl", "rb") as f:
    models = pickle.load(f)

ours = pd.read_csv(REPO/"all_workflows.csv")
on = pd.DataFrame({"workflow":ours["workflow"], "process":ours["process"],
    "a":pd.to_numeric(ours["a_bytes"],errors="coerce"),
    "c":pd.to_numeric(ours["c_bytes"],errors="coerce"),
    "M":pd.to_numeric(ours["M_peak_rss_bytes"],errors="coerce"),
}).dropna(subset=["a","M"]).query("M>0 and a>=0").reset_index(drop=True)
on = on[on["workflow"] != "methylseq"]
sys.path.insert(0, str(REPO))
from load_pyradiomics import load_pyradiomics
pn = load_pyradiomics(str(REPO/"pyradiomics_32k.csv"))[["workflow","process","a","c","M"]]
df = pd.concat([on, pn], ignore_index=True)

test_rows = []
for (wf, proc), grp in df.groupby(["workflow","process"]):
    if len(grp) < MIN: continue
    g = grp.sample(frac=1, random_state=SEED).reset_index(drop=True)
    cut = int(len(g) * 0.8)
    te = g.iloc[cut:].copy(); te["bucket"] = f"{wf}::{proc}"
    test_rows.append(te)
test_df = pd.concat(test_rows, ignore_index=True)

def predict(model_key):
    """Return (actual_M, predicted_M) for one model+variant across all buckets."""
    actuals, preds = [], []
    for bucket, g in test_df.groupby("bucket"):
        info = models.get(bucket)
        if info is None: continue
        m_obj = info["fits"].get(model_key)
        if m_obj is None: continue
        a_log = np.log(g["a"].values + EPS)
        c_present = g["c"].notna().values
        if "_joint" in model_key:
            if not c_present.any(): continue
            a_log_c = a_log[c_present]
            c_log_c = np.log(g["c"].values[c_present] + EPS)
            X = np.column_stack([a_log_c, c_log_c])
            actual = g["M"].values[c_present]
        else:
            X = a_log.reshape(-1, 1)
            actual = g["M"].values
        try:
            from ngboost import NGBRegressor
            if isinstance(m_obj, NGBRegressor):
                p_log = np.log(np.maximum(1.0, m_obj.pred_dist(X).mean()))
            else:
                p_log = m_obj.predict(X)
        except Exception:
            continue
        actuals.append(actual)
        preds.append(np.exp(p_log))
    return np.concatenate(actuals), np.concatenate(preds)

MODEL_NAMES = [("lr", "Linear Regression"), ("knn", "KNN (k=5)"),
               ("mlp", "MLP (64-32)"), ("rf", "Random Forest"),
               ("lgbm", "LightGBM"), ("ngb", "NGBoost LogNormal")]

# ====== FIGURE 9 — 2 x 6 grid of per-model calibration scatters ======
fig, axes = plt.subplots(2, 6, figsize=(28, 11), sharex=True, sharey=True)
for col, (key, label) in enumerate(MODEL_NAMES):
    for row, variant in enumerate(["sizey", "joint"]):
        ax = axes[row, col]
        try:
            M, P = predict(f"{key}_{variant}")
            mape = float(np.mean(np.abs(P - M) / M) * 100)
            color = "#6c757d" if variant == "sizey" else "#16a085"
            ax.scatter(M / 1024**2, P / 1024**2, alpha=0.35, s=70,
                       color=color, edgecolor="white", linewidth=0.5)
            lo = min(M.min(), P.min()) / 1024**2
            hi = max(M.max(), P.max()) / 1024**2
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=2.5, label="perfect")
            ax.plot([lo, hi], [2 * lo, 2 * hi], "r:", linewidth=1.8, alpha=0.7, label="±2×")
            ax.plot([lo, hi], [0.5 * lo, 0.5 * hi], "r:", linewidth=1.8, alpha=0.7)
            ax.text(0.05, 0.92, f"MAPE = {mape:.2f}%", transform=ax.transAxes,
                    fontsize=15, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="black", linewidth=1.2))
        except Exception as e:
            ax.text(0.5, 0.5, f"(no fit)\n{str(e)[:30]}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        if row == 0:
            ax.set_title(label, fontsize=20)
        if col == 0:
            ax.set_ylabel(("Sizey  (a only)" if variant == "sizey" else "Joint  (a, c)"),
                          fontsize=18, fontweight="bold")
        if row == 1:
            ax.set_xlabel("Actual M (MB)", fontsize=16)
fig.text(0.06, 0.5, "Predicted M (MB)", rotation=90, fontsize=22, va="center")
fig.suptitle("Per-model calibration  —  predicted vs actual peak RSS  (held-out test)",
             fontsize=26, y=0.995)
fig.tight_layout(rect=[0.07, 0, 1, 0.96])
fig.savefig(REPO / "figures/fig9_per_model_calibration.png")
plt.close()
print("wrote fig9_per_model_calibration.png")

# ====== FIGURE 10 — Error % vs actual-M magnitude, one line per model ======
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Panel: bin by actual M, plot mean |relative error| per bin, one line per model
NUM_BINS = 25
COLORS = {"lr":"#377eb8", "knn":"#4daf4a", "mlp":"#984ea3",
          "rf":"#ff7f00", "lgbm":"#e41a1c", "ngb":"#a65628"}
MARKERS = {"lr":"o", "knn":"s", "mlp":"D", "rf":"^", "lgbm":"P", "ngb":"X"}

for ax, variant, title in [
    (axes[0], "sizey", "Sizey  variant  (X = log a)"),
    (axes[1], "joint", "Joint  variant  (X = log a, log c)"),
]:
    for key, label in MODEL_NAMES:
        try:
            M, P = predict(f"{key}_{variant}")
            err_pct = np.abs(P - M) / M * 100
            order = np.argsort(M)
            M_sorted = M[order]; e_sorted = err_pct[order]
            edges = np.percentile(M_sorted, np.linspace(0, 100, NUM_BINS + 1))
            edges = np.unique(edges)
            mids, mean_err = [], []
            for i in range(len(edges) - 1):
                mask = (M_sorted >= edges[i]) & (M_sorted <= edges[i+1])
                if mask.sum() < 5: continue
                mids.append(np.median(M_sorted[mask]))
                mean_err.append(np.mean(e_sorted[mask]))
            ax.plot(np.array(mids) / 1024**2, mean_err,
                    marker=MARKERS[key], color=COLORS[key], label=label.split()[0],
                    markeredgecolor="black", markeredgewidth=1.3, alpha=0.9)
        except Exception:
            continue
    ax.set_xscale("log")
    ax.set_xlabel("Actual peak RSS  (MB)")
    ax.set_ylabel("Mean |error|  (%)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=15, ncol=2, loc="upper right")
fig.suptitle("Per-model prediction error vs task size  (held-out test, binned by actual M)",
             y=1.02, fontsize=24)
fig.tight_layout()
fig.savefig(REPO / "figures/fig10_error_vs_magnitude.png")
plt.close()
print("wrote fig10_error_vs_magnitude.png")

print("\nDone — both figures in /shared/training/figures/")
