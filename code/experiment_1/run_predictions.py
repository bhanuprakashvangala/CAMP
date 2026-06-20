"""
Join all datasets — keeping ALL columns — and run every trained model on every row.

Output:
  analysis/combined_dataset.csv     -- joined raw data (all original columns preserved)
  analysis/predictions_all.csv      -- combined data + 12 model predictions per row
  analysis/per_row_errors.csv       -- per-row absolute % error per model
  analysis/summary_per_bucket.csv   -- aggregate by bucket
  analysis/verify_models_used.txt   -- proof that the trained pkl was actually used
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, sys, os
import numpy as np
import pandas as pd
from pathlib import Path

REPO   = Path(__file__).resolve().parent.parent
ANALY  = Path(__file__).resolve().parent
EPS, K = 1.0, 1.5

# ---- 1. Load all 3 sources, KEEP ALL COLUMNS ----
print("Loading datasets...")

# Source A: ours (nf-core + iwd, joined by rebuild_datasets.py)
ours = pd.read_csv(REPO/"output"/"joined"/"all_workflows.csv")
ours["source"] = "ours"
# Add convenience canonical columns (a, c, M, runtime) so the predictor can find them uniformly
ours["a"]       = pd.to_numeric(ours["a_bytes"], errors="coerce")
ours["c"]       = pd.to_numeric(ours["c_bytes"], errors="coerce")
ours["M"]       = pd.to_numeric(ours["M_peak_rss_bytes"], errors="coerce")
ours["runtime"] = pd.to_numeric(ours["runtime_seconds"], errors="coerce")
ours = ours[(ours["a"]>=0) & (ours["M"]>0)].dropna(subset=["a","M"]).reset_index(drop=True)
print(f"  ours (nf-core + iwd): {len(ours):,} rows × {len(ours.columns)} cols")

# Source B: pyradiomics 32k — keep ALL the rich features
pyr = pd.read_csv(REPO.parent/"task_metrics_pyradiomics_32k_detailed.csv")
pyr["source"] = "pyradiomics"
pyr["workflow"] = "pyradiomics"
proc = pyr["task_type"].astype(str)
fc   = pyr["feature_class"].fillna("").astype(str)
pyr["process"] = np.where((proc == "extract_feature_class") & (fc != ""), proc + "::" + fc, proc)
pyr["a"]       = pd.to_numeric(pyr["input_file_size_bytes"], errors="coerce")
pyr["c"]       = pd.to_numeric(pyr["rchar"], errors="coerce")
pyr["M"]       = pd.to_numeric(pyr["peak_rss_kb"], errors="coerce") * 1024.0
pyr["runtime"] = pd.to_numeric(pyr["runtime_seconds"], errors="coerce")
pyr = pyr[(pyr["a"]>=0) & (pyr["c"]>0) & (pyr["M"]>0)].dropna(subset=["a","c","M"]).reset_index(drop=True)
print(f"  pyradiomics: {len(pyr):,} rows × {len(pyr.columns)} cols")

# Source C: trace_methylseq (sizey published trace, has many extra columns)
trace = pd.read_csv(REPO.parent/"trace_methylseq.csv")
trace["source"]   = "trace_methylseq"
trace["workflow"] = "methylseq_naga"
trace["a"]       = pd.to_numeric(trace["read_bytes"], errors="coerce")
trace["c"]       = pd.to_numeric(trace["rchar"], errors="coerce")
trace["M"]       = pd.to_numeric(trace["peak_rss"], errors="coerce")
trace["runtime"] = pd.to_numeric(trace["realtime"], errors="coerce") / 1000.0
trace = trace[(trace["a"]>=0) & (trace["c"]>0) & (trace["M"]>0)].dropna(subset=["a","c","M"]).reset_index(drop=True)
print(f"  trace_methylseq: {len(trace):,} rows × {len(trace.columns)} cols")

# Outer-merge — all columns from all 3 sources end up here.
# Where a source doesn't have a given column, NaN appears. That's exactly what the user asked for.
df = pd.concat([ours, pyr, trace], ignore_index=True, sort=False)
df["runtime"] = df["runtime"].fillna(60.0)
print(f"\n  COMBINED: {len(df):,} rows × {len(df.columns)} cols")
print(df.groupby(["source","workflow"]).size().rename("n").to_string())

df.to_csv(ANALY/"combined_dataset.csv", index=False)
print(f"\nwrote {ANALY/'combined_dataset.csv'}  ({(ANALY/'combined_dataset.csv').stat().st_size:,} bytes)")

# ---- 2. Load the trained model bundle ----
print("\nLoading trained model bundle (per_task_paper.pkl)...")
with open(REPO/"models_unified"/"per_task_paper.pkl", "rb") as f:
    models = pickle.load(f)
print(f"  buckets in trained model: {len(models)}")
print(f"  bucket keys (first 5): {list(models.keys())[:5]}")

MODEL_NAMES = ["lr", "knn", "mlp", "rf", "lgbm", "ngb"]

# ---- 2b. Train GLOBAL fallback (only used for unseen buckets) ----
print("\nTraining global fallback for unseen buckets...")
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
try:
    from ngboost import NGBRegressor
    from ngboost.distns import LogNormal
    HAS_NGB = True
except ImportError:
    HAS_NGB = False

a_log_all  = np.log(df["a"].values + EPS)
M_log_all  = np.log(df["M"].values + EPS)
mask_c_all = df["c"].notna().values
a_log_c    = a_log_all[mask_c_all]
c_log_c    = np.log(df["c"].values[mask_c_all] + EPS)
M_log_c    = M_log_all[mask_c_all]

def mk(name):
    if name == "lr":   return LinearRegression()
    if name == "knn":  return Pipeline([("s",StandardScaler()),("m",KNeighborsRegressor(n_neighbors=5))])
    if name == "mlp":  return Pipeline([("s",StandardScaler()),("m",MLPRegressor(hidden_layer_sizes=(64,32),max_iter=400,random_state=42))])
    if name == "rf":   return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    if name == "lgbm": return LGBMRegressor(n_estimators=200,learning_rate=0.05,num_leaves=31,min_data_in_leaf=2,verbose=-1)
    if name == "ngb" and HAS_NGB:
        return NGBRegressor(Dist=LogNormal,n_estimators=200,learning_rate=0.05,verbose=False,random_state=42)
    return None

global_fits = {}
for name in MODEL_NAMES:
    m_s = mk(name)
    if m_s is not None:
        try:
            if name == "ngb": m_s.fit(a_log_all.reshape(-1,1), np.exp(M_log_all))
            else:             m_s.fit(a_log_all.reshape(-1,1), M_log_all)
            preds = (np.log(np.maximum(1.0, m_s.pred_dist(a_log_all.reshape(-1,1)).mean()))
                     if name=="ngb" else m_s.predict(a_log_all.reshape(-1,1)))
            sigma_s = float(np.std(M_log_all - preds))
            global_fits[f"{name}_sizey"] = (m_s, sigma_s)
        except Exception: pass
    m_j = mk(name)
    if m_j is not None:
        try:
            X = np.column_stack([a_log_c, c_log_c])
            if name == "ngb": m_j.fit(X, np.exp(M_log_c))
            else:             m_j.fit(X, M_log_c)
            preds = (np.log(np.maximum(1.0, m_j.pred_dist(X).mean())) if name=="ngb" else m_j.predict(X))
            sigma_j = float(np.std(M_log_c - preds))
            global_fits[f"{name}_joint"] = (m_j, sigma_j)
        except Exception: pass
print(f"  global fallback fits: {len(global_fits)}")

def predict_one(model_obj, X):
    try:
        from ngboost import NGBRegressor
        if isinstance(model_obj, NGBRegressor):
            return np.log(np.maximum(1.0, model_obj.pred_dist(X).mean()))
    except ImportError: pass
    return model_obj.predict(X)

# ---- 3. Predict per row ----
print("\nRunning predictions...")
out_cols = {}
for name in MODEL_NAMES:
    for var in ["sizey","joint"]:
        out_cols[f"pred_{name}_{var}_MB"] = np.full(len(df), np.nan)
        out_cols[f"safe_{name}_{var}_MB"] = np.full(len(df), np.nan)
strategy = np.full(len(df), "", dtype=object)

for (wf, proc), g in df.groupby(["workflow", "process"]):
    key = f"{wf}::{proc}"
    info = models.get(key)
    idxs = g.index.values
    a_log = np.log(g["a"].values + EPS)
    c_present = g["c"].notna().values
    a_log_c_b = a_log[c_present]
    c_log_c_b = np.log(g["c"].values[c_present] + EPS) if c_present.any() else np.array([])

    if info is None:                                  # ── Global fallback path ──
        strategy[idxs] = "global_fallback"
        for name in MODEL_NAMES:
            if f"{name}_sizey" in global_fits:
                m_s, sg_s = global_fits[f"{name}_sizey"]
                try:
                    p_log = predict_one(m_s, a_log.reshape(-1,1))
                    out_cols[f"pred_{name}_sizey_MB"][idxs] = np.exp(p_log)/1024**2
                    out_cols[f"safe_{name}_sizey_MB"][idxs] = np.exp(p_log+K*sg_s)/1024**2
                except Exception: pass
            if f"{name}_joint" in global_fits and c_present.any():
                m_j, sg_j = global_fits[f"{name}_joint"]
                try:
                    p_log = predict_one(m_j, np.column_stack([a_log_c_b, c_log_c_b]))
                    sub = idxs[c_present]
                    out_cols[f"pred_{name}_joint_MB"][sub] = np.exp(p_log)/1024**2
                    out_cols[f"safe_{name}_joint_MB"][sub] = np.exp(p_log+K*sg_j)/1024**2
                except Exception: pass
        continue

    # ── Bucket-specific path: this is where TRAINED model from per_task_paper.pkl is used ──
    strategy[idxs] = "bucket_specific"
    for name in MODEL_NAMES:
        m_s = info["fits"].get(f"{name}_sizey")
        sg_s = info["sigmas"].get(f"{name}_sizey", 0.5)
        if m_s is not None:
            try:
                p_log = predict_one(m_s, a_log.reshape(-1,1))
                out_cols[f"pred_{name}_sizey_MB"][idxs] = np.exp(p_log)/1024**2
                out_cols[f"safe_{name}_sizey_MB"][idxs] = np.exp(p_log+K*sg_s)/1024**2
            except Exception: pass
        m_j = info["fits"].get(f"{name}_joint")
        sg_j = info["sigmas"].get(f"{name}_joint", 0.5)
        if m_j is not None and c_present.any():
            try:
                p_log = predict_one(m_j, np.column_stack([a_log_c_b, c_log_c_b]))
                sub = idxs[c_present]
                out_cols[f"pred_{name}_joint_MB"][sub] = np.exp(p_log)/1024**2
                out_cols[f"safe_{name}_joint_MB"][sub] = np.exp(p_log+K*sg_j)/1024**2
            except Exception: pass

for k, v in out_cols.items(): df[k] = v
df["prediction_strategy"] = strategy

# Add MB convenience copies (so users don't have to / 1024**2 mentally)
df["a_MB"] = df["a"] / 1024**2
df["c_MB"] = df["c"] / 1024**2
df["M_MB"] = df["M"] / 1024**2

# Reorder: identifiers first, then a/c/M, then predictions, then everything else
front = ["source", "workflow", "process", "prediction_strategy",
         "a_MB", "c_MB", "M_MB", "runtime", "a", "c", "M"]
pred_cols = [c for c in df.columns if c.startswith("pred_") or c.startswith("safe_")]
other_cols = [c for c in df.columns if c not in front + pred_cols]
df = df[front + pred_cols + other_cols]

df.to_csv(ANALY/"predictions_all.csv", index=False)
print(f"wrote {ANALY/'predictions_all.csv'}  ({len(df):,} rows × {len(df.columns)} cols, "
      f"{(ANALY/'predictions_all.csv').stat().st_size:,} bytes)")

# ---- 4. Per-row errors ----
err = df[["source","workflow","process","prediction_strategy","M_MB"]].copy()
for name in MODEL_NAMES:
    for var in ["sizey","joint"]:
        col = f"pred_{name}_{var}_MB"
        if col in df.columns:
            err[f"err_{name}_{var}_pct"] = np.abs(df[col] - df["M_MB"]) / df["M_MB"] * 100
err.to_csv(ANALY/"per_row_errors.csv", index=False)
print(f"wrote {ANALY/'per_row_errors.csv'}")

# ---- 5. Per-bucket aggregate ----
agg = []
for (wf, proc), g in df.groupby(["workflow","process"]):
    row = {"workflow":wf, "process":proc, "n":len(g),
           "n_c":int(g["c"].notna().sum()),
           "strategy":g["prediction_strategy"].iloc[0]}
    for name in MODEL_NAMES:
        for var in ["sizey","joint"]:
            col = f"pred_{name}_{var}_MB"
            if col in g.columns and g[col].notna().any():
                ape = np.abs(g[col] - g["M_MB"])/g["M_MB"] * 100
                row[f"{name}_{var}_MAPE"] = float(np.mean(ape.dropna()))
    agg.append(row)
pd.DataFrame(agg).to_csv(ANALY/"summary_per_bucket.csv", index=False)
print(f"wrote {ANALY/'summary_per_bucket.csv'}")

# ---- 6. PROOF that the trained models were used: replicate one prediction ----
verify_lines = []
verify_lines.append("="*100)
verify_lines.append("VERIFICATION — proof that per_task_paper.pkl was actually used")
verify_lines.append("="*100)

# Pick a pyradiomics row that should have used the bucket-specific RF Joint
target_bucket = "pyradiomics::extract_feature_class::glcm"
sample = df[(df["workflow"]+"::"+df["process"] == target_bucket) &
            (df["prediction_strategy"]=="bucket_specific") &
            (df["c"].notna())].iloc[0]
a, c, M_actual = float(sample["a"]), float(sample["c"]), float(sample["M"])
csv_pred_rf_joint = float(sample["pred_rf_joint_MB"])

# Manually load the model and reproduce
info = models[target_bucket]
rf_joint_model = info["fits"]["rf_joint"]
sigma_rf_joint = info["sigmas"]["rf_joint"]
X = np.array([[np.log(a + EPS), np.log(c + EPS)]])
p_log = rf_joint_model.predict(X)[0]
manual_pred_MB = np.exp(p_log) / 1024**2
manual_safe_MB = np.exp(p_log + K * sigma_rf_joint) / 1024**2

verify_lines.append(f"\nTest row from bucket: {target_bucket}")
verify_lines.append(f"  a (input file size) = {a:>15.0f} bytes  ({a/1024**2:.2f} MB)")
verify_lines.append(f"  c (bytes consumed)  = {c:>15.0f} bytes  ({c/1024**2:.2f} MB)")
verify_lines.append(f"  M (actual peak RSS) = {M_actual:>15.0f} bytes  ({M_actual/1024**2:.2f} MB)")
verify_lines.append(f"")
verify_lines.append(f"Prediction in CSV (pred_rf_joint_MB):  {csv_pred_rf_joint:.4f} MB")
verify_lines.append(f"Manual replay from pkl + RandomForest: {manual_pred_MB:.4f} MB")
verify_lines.append(f"Match: {'YES' if abs(csv_pred_rf_joint - manual_pred_MB) < 0.01 else 'NO'}")
verify_lines.append(f"")
verify_lines.append(f"Model details from pkl:")
verify_lines.append(f"  rf_joint type       = {type(rf_joint_model).__name__}")
verify_lines.append(f"  rf_joint trees      = {rf_joint_model.n_estimators}")
verify_lines.append(f"  sg_rf_joint (cushion)= {sigma_rf_joint:.6f}  (log space)")
verify_lines.append(f"  bucket n            = {info['n_tr']} train + {info['n_te']} test")
verify_lines.append(f"")
verify_lines.append("All 6 model classes × 2 variants for this single row:")
for name in MODEL_NAMES:
    for var in ["sizey", "joint"]:
        m_obj = info["fits"].get(f"{name}_{var}")
        if m_obj is None:
            verify_lines.append(f"  {name}_{var:<6} -- (not fitted in this bucket)")
            continue
        if var == "sizey":
            X = np.array([[np.log(a + EPS)]])
        else:
            X = np.array([[np.log(a + EPS), np.log(c + EPS)]])
        try:
            from ngboost import NGBRegressor
            if isinstance(m_obj, NGBRegressor):
                p = float(np.log(np.maximum(1.0, m_obj.pred_dist(X).mean()))[0])
            else:
                p = float(m_obj.predict(X)[0])
        except Exception:
            verify_lines.append(f"  {name}_{var:<6}  -- (predict failed)"); continue
        pred_MB = float(np.exp(p) / 1024**2)
        verify_lines.append(f"  {name}_{var:<6}  pred = {pred_MB:>9.2f} MB  "
                            f"(M_actual = {M_actual/1024**2:.2f} MB, "
                            f"err = {abs(pred_MB - M_actual/1024**2)/(M_actual/1024**2)*100:>5.2f}%)")

(ANALY/"verify_models_used.txt").write_text("\n".join(verify_lines))
print("\n" + "\n".join(verify_lines))

# ---- 7. Final coverage report ----
print("\n" + "="*100)
print("FINAL COVERAGE")
print("="*100)
print(f"Total rows:                 {len(df):,}")
print(f"Bucket-specific predictions: {(df['prediction_strategy']=='bucket_specific').sum():,}  (used per_task_paper.pkl)")
print(f"Global-fallback predictions: {(df['prediction_strategy']=='global_fallback').sum():,}  (newly fit)")
print(f"Columns in predictions_all.csv: {len(df.columns)}  (all original features preserved)")

print("\nFILES:")
for f in sorted(ANALY.glob("*.csv")) + sorted(ANALY.glob("*.txt")) + sorted(ANALY.glob("*.py")):
    print(f"  {f.name:<35}  {f.stat().st_size:>12,} bytes")
