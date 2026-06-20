"""Simple/standard ML metrics for the 3 headline models on the held-out set."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd, sys
from pathlib import Path
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix)
from scipy.stats import pearsonr, spearmanr

REPO = Path("/shared/training")
EPS, K, MIN, SEED = 1.0, 1.5, 10, 42

# Reload pickled trained model dict from the paper notebook
with open(REPO/"models_unified/per_task_paper.pkl", "rb") as f:
    models = pickle.load(f)

# ---- Reproduce the same train/test split as the paper notebook ----
ours = pd.read_csv(REPO/"all_workflows.csv")
on = pd.DataFrame({"workflow":ours["workflow"], "process":ours["process"],
    "a":pd.to_numeric(ours["a_bytes"],errors="coerce"),
    "c":pd.to_numeric(ours["c_bytes"],errors="coerce"),
    "M":pd.to_numeric(ours["M_peak_rss_bytes"],errors="coerce"),
    "runtime":pd.to_numeric(ours["runtime_seconds"],errors="coerce"),
}).dropna(subset=["a","M"]).query("M>0 and a>=0").reset_index(drop=True)
on = on[on["workflow"] != "methylseq"]
sys.path.insert(0, str(REPO))
from load_pyradiomics import load_pyradiomics
pn = load_pyradiomics(str(REPO/"pyradiomics_32k.csv"))[["workflow","process","a","c","M","runtime_sec"]].rename(columns={"runtime_sec":"runtime"})
df = pd.concat([on, pn], ignore_index=True)
df["runtime"] = df["runtime"].fillna(60.0)

# Re-create test split per bucket
test_rows = []
for (wf, proc), grp in df.groupby(["workflow","process"]):
    if len(grp) < MIN: continue
    g = grp.sample(frac=1, random_state=SEED).reset_index(drop=True)
    cut = int(len(g) * 0.8)
    te = g.iloc[cut:].copy(); te["bucket"] = f"{wf}::{proc}"
    test_rows.append(te)
test_df = pd.concat(test_rows, ignore_index=True)

# ---- For each (bucket, model_name), reproduce predictions on test ----
def predict_all(model_key):
    """Walk the test rows, get predicted log M for the chosen model+variant."""
    preds, actuals, runtime, bucket_med_M, c_present_flags = [], [], [], [], []
    for bucket, g in test_df.groupby("bucket"):
        info = models.get(bucket)
        if info is None: continue
        m_obj = info["fits"].get(model_key)
        sigma = info["sigmas"].get(model_key)
        if m_obj is None or sigma is None: continue
        a_log = np.log(g["a"].values + EPS)
        c_present = g["c"].notna().values
        # bucket median M (from training half) for high-memory class label
        # We use overall test M as ref (consistent across models)
        bucket_med = float(np.median(g["M"].values))
        if "_joint" in model_key:
            if not c_present.any(): continue
            a_log_c = a_log[c_present]
            c_log_c = np.log(g["c"].values[c_present] + EPS)
            X = np.column_stack([a_log_c, c_log_c])
            try:
                from ngboost import NGBRegressor
                if isinstance(m_obj, NGBRegressor):
                    pred = np.log(np.maximum(1.0, m_obj.pred_dist(X).mean()))
                else:
                    pred = m_obj.predict(X)
            except Exception:
                pred = m_obj.predict(X)
            sub_mask = c_present
        else:
            X = a_log.reshape(-1,1)
            try:
                from ngboost import NGBRegressor
                if isinstance(m_obj, NGBRegressor):
                    pred = np.log(np.maximum(1.0, m_obj.pred_dist(X).mean()))
                else:
                    pred = m_obj.predict(X)
            except Exception:
                pred = m_obj.predict(X)
            sub_mask = np.ones(len(g), dtype=bool)
        M_actual = g["M"].values[sub_mask]
        rt = g["runtime"].values[sub_mask]
        preds.append(pred); actuals.append(M_actual); runtime.append(rt)
        bucket_med_M.extend([bucket_med]*len(M_actual))
    if not preds:
        return None
    return (np.concatenate(preds), np.concatenate(actuals),
            np.concatenate(runtime), np.array(bucket_med_M))

def all_metrics(name, p_log, M, rt, bucket_med):
    """Compute regression + classification metrics."""
    P = np.exp(p_log)                  # predicted M (raw bytes)
    safe = np.exp(p_log + K * 0.1)     # we use a single common cushion for fair compare; sigma differs per bucket
    # but for safe-alloc we want bucket-specific sigma — use 0 for clean comparison
    # Use predicted directly to be honest:
    M_safe = P  # safe = predicted (no cushion) for raw accuracy comparison

    # ---- Regression metrics ----
    err = P - M
    abs_pct = np.abs(err) / M * 100
    metrics = {
        "model":          name,
        "n":              len(M),
        "MAE_MB":         mean_absolute_error(M/1024**2, P/1024**2),
        "RMSE_MB":        np.sqrt(mean_squared_error(M/1024**2, P/1024**2)),
        "MAPE_pct":       float(np.mean(abs_pct)),
        "MedAPE_pct":     float(np.median(abs_pct)),
        "Pearson_r":      float(pearsonr(M, P)[0]),
        "Spearman_rho":   float(spearmanr(M, P)[0]),
        "R2_log":         float(1 - np.sum((np.log(M)-p_log)**2) /
                                np.sum((np.log(M)-np.log(M).mean())**2)),
        # ---- Tolerance bands (= "accuracy at tolerance") ----
        "Acc_within_10pct":  float(np.mean(abs_pct <= 10) * 100),
        "Acc_within_25pct":  float(np.mean(abs_pct <= 25) * 100),
        "Acc_within_50pct":  float(np.mean(abs_pct <= 50) * 100),
        "Acc_within_2x":     float(np.mean((P >= 0.5*M) & (P <= 2.0*M)) * 100),
        # ---- OOM viewpoint (allocation = predicted, no cushion for fair compare) ----
        "OOM_rate_pct":      float(np.mean(P < M) * 100),
        # ---- Binary classification: "high-memory task"  (M > bucket median) ----
        "Precision_HM":   None,
        "Recall_HM":      None,
        "F1_HM":          None,
        "Accuracy_HM":    None,
    }
    y_true = (M > bucket_med).astype(int)
    y_pred = (P > bucket_med).astype(int)
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        metrics["Precision_HM"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["Recall_HM"]    = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["F1_HM"]        = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["Accuracy_HM"]  = float(accuracy_score(y_true, y_pred))
    return metrics

# ===== Compute for the 3 headline models =====
candidates = [("LGBM Sizey", "lgbm_sizey"),
              ("LGBM Joint", "lgbm_joint"),
              ("RF Joint",   "rf_joint")]

rows = []
for label, key in candidates:
    out = predict_all(key)
    if out is None:
        print(f"skip {label}"); continue
    p, M, rt, bm = out
    m = all_metrics(label, p, M, rt, bm)
    rows.append(m)

# ===== Sizey ensemble (per-bucket best of LR/KNN/MLP/RF) =====
def predict_sizey_ensemble():
    preds, actuals, runtime, med = [], [], [], []
    chosen_per_bucket = {}
    for bucket, g in test_df.groupby("bucket"):
        info = models.get(bucket)
        if info is None: continue
        # pick best baseline by per-bucket sizey-MAPE on test (replicates Sizey RAQ selection)
        best_key, best_mape = None, np.inf
        a_log = np.log(g["a"].values + EPS)
        for cand in ["lr","knn","mlp","rf"]:
            mk = f"{cand}_sizey"
            mo = info["fits"].get(mk)
            if mo is None: continue
            try:
                from ngboost import NGBRegressor
                if isinstance(mo, NGBRegressor):
                    p = np.log(np.maximum(1.0, mo.pred_dist(a_log.reshape(-1,1)).mean()))
                else:
                    p = mo.predict(a_log.reshape(-1,1))
            except Exception:
                continue
            mp = float(np.mean(np.abs(np.exp(p) - g["M"].values) / g["M"].values) * 100)
            if mp < best_mape: best_mape, best_key = mp, mk
        if best_key is None: continue
        chosen_per_bucket[bucket] = best_key
        mo = info["fits"][best_key]
        try:
            from ngboost import NGBRegressor
            if isinstance(mo, NGBRegressor):
                p = np.log(np.maximum(1.0, mo.pred_dist(a_log.reshape(-1,1)).mean()))
            else:
                p = mo.predict(a_log.reshape(-1,1))
        except Exception:
            continue
        bucket_med = float(np.median(g["M"].values))
        preds.append(p); actuals.append(g["M"].values)
        runtime.append(g["runtime"].values); med.extend([bucket_med]*len(g))
    return (np.concatenate(preds), np.concatenate(actuals),
            np.concatenate(runtime), np.array(med), chosen_per_bucket)

p, M, rt, bm, choices = predict_sizey_ensemble()
m = all_metrics("Sizey Ensemble", p, M, rt, bm)
rows.insert(0, m)
print("Sizey ensemble per-bucket selection:", {k:v for k,v in choices.items()})

# ===== Render table =====
metric_df = pd.DataFrame(rows)
cols_order = ["model","n","MAE_MB","RMSE_MB","MAPE_pct","MedAPE_pct",
              "Pearson_r","Spearman_rho","R2_log",
              "Acc_within_10pct","Acc_within_25pct","Acc_within_50pct","Acc_within_2x",
              "OOM_rate_pct","Precision_HM","Recall_HM","F1_HM","Accuracy_HM"]
metric_df = metric_df[cols_order]

print()
print("="*120)
print("SIMPLE METRICS — Held-out test set")
print("="*120)
for _, r in metric_df.iterrows():
    print(f"\n--- {r['model']:<20}  n={int(r['n'])} ---")
    print(f"  MAE          = {r['MAE_MB']:>9.2f} MB        (mean absolute error in megabytes)")
    print(f"  RMSE         = {r['RMSE_MB']:>9.2f} MB        (root mean squared error)")
    print(f"  MAPE         = {r['MAPE_pct']:>9.2f} %         (mean absolute percentage error)")
    print(f"  MedAPE       = {r['MedAPE_pct']:>9.2f} %         (median APE — robust to outliers)")
    print(f"  Pearson r    = {r['Pearson_r']:>9.4f}           (linear corr predicted vs actual)")
    print(f"  Spearman ρ   = {r['Spearman_rho']:>9.4f}           (rank corr predicted vs actual)")
    print(f"  R² (log)     = {r['R2_log']:>9.4f}           (variance explained in log-M)")
    print(f"  Accuracy:")
    print(f"    within 10%  = {r['Acc_within_10pct']:>5.2f} %")
    print(f"    within 25%  = {r['Acc_within_25pct']:>5.2f} %")
    print(f"    within 50%  = {r['Acc_within_50pct']:>5.2f} %")
    print(f"    within 2x   = {r['Acc_within_2x']:>5.2f} %    (factor of two)")
    print(f"  OOM rate (raw, no cushion) = {r['OOM_rate_pct']:>5.2f} %")
    print(f"  Binary class 'high-mem task' (M > bucket median):")
    if r['Precision_HM'] is not None:
        print(f"    Precision   = {r['Precision_HM']:>6.4f}")
        print(f"    Recall      = {r['Recall_HM']:>6.4f}")
        print(f"    F1          = {r['F1_HM']:>6.4f}")
        print(f"    Accuracy    = {r['Accuracy_HM']:>6.4f}")

# ===== Compact comparison table =====
print()
print("="*120)
print("COMPACT COMPARISON")
print("="*120)
short = metric_df[["model","MAE_MB","RMSE_MB","MAPE_pct","Pearson_r","R2_log",
                   "Acc_within_25pct","Acc_within_2x","OOM_rate_pct",
                   "Precision_HM","Recall_HM","F1_HM","Accuracy_HM"]]
print(short.round(3).to_string(index=False))
short.to_csv(REPO/"figures/simple_metrics.csv", index=False)
print()
print(f"saved: {REPO}/figures/simple_metrics.csv")
