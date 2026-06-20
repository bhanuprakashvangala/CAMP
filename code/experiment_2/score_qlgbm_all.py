#!/usr/bin/env python3
"""
Score every row of the combined eBPF dataset with both Q-LightGBM Sizey (a only)
and Joint (a, c) (per-bucket if available; global fallback otherwise) and emit:

  workflow, process, task_hash, a_MB, c_MB,
  pred_qlgbm_sizey_MB, std_qlgbm_sizey_MB,
  q50_qlgbm_sizey_MB, q95_qlgbm_sizey_MB, safe_qlgbm_sizey_MB,
  pred_qlgbm_joint_MB, std_qlgbm_joint_MB,
  q50_qlgbm_joint_MB, q95_qlgbm_joint_MB, safe_qlgbm_joint_MB,
  proxy_pred_M_MB, proxy_intercept,
  proxy_log_c_coef, proxy_log_a_coef, proxy_log_ca_ratio_coef,
  M_MB

Output: predictions_qlgbm_exp2_all.csv
"""

from __future__ import annotations
from pathlib import Path
import pickle, math, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "ebpf data"
EPS = 1.0
Z95 = 1.6448536269514722  # one-sided 95% normal quantile

# --- loaders (mirror build_exp2_notebook.py) -------------------------------

def parse_size_to_bytes(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    if s.endswith(" KB"): return float(s[:-3]) * 1024
    if s.endswith(" MB"): return float(s[:-3]) * 1024**2
    if s.endswith(" GB"): return float(s[:-3]) * 1024**3
    if s.endswith(" TB"): return float(s[:-3]) * 1024**4
    if s.endswith(" B"):  return float(s[:-2])
    try: return float(s)
    except: return np.nan

def parse_runtime(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    if s.endswith("ms"):
        try: return float(s[:-2]) / 1000.0
        except: return np.nan
    if s.endswith("s"):
        try: return float(s[:-1])
        except: return np.nan
    if s.endswith("m"):
        try: return float(s[:-1]) * 60
        except: return np.nan
    try: return float(s)
    except: return np.nan

def load_string_fmt(path, workflow_name):
    df_ = pd.read_csv(path)
    df_["workflow"] = workflow_name
    df_["a"] = df_["trace_rchar"].apply(parse_size_to_bytes)
    df_["c"] = pd.to_numeric(df_["ebpf_total_bytes"], errors="coerce")
    df_["M"] = df_["trace_peak_rss"].apply(parse_size_to_bytes)
    df_["runtime_sec"] = df_["trace_realtime"].apply(parse_runtime)
    return df_[["workflow","process","task_hash","a","c","M","runtime_sec"]]


def load_all() -> pd.DataFrame:
    m1 = load_string_fmt(DATA/"Minimap_merged_audit_trace_2.csv", "minimap2_audit_nf")
    m2 = pd.read_csv(DATA/"task_metrics_with_ebpf_mcmicro.csv")
    m2["workflow"] = "mcmicro"
    m2["a"] = pd.to_numeric(m2["trace_rchar"], errors="coerce")
    m2["c"] = pd.to_numeric(m2["ebpf_total_bytes"], errors="coerce")
    m2["M"] = pd.to_numeric(m2["trace_peak_rss_kb"], errors="coerce") * 1024.0
    m2["runtime_sec"] = pd.to_numeric(m2["trace_realtime_ms"], errors="coerce") / 1000.0
    m2 = m2[["workflow","process","task_hash","a","c","M","runtime_sec"]]
    m3 = load_string_fmt(DATA/"Bowtie_merged_audit_trace.csv", "bowtie2_audit_nf")
    df = pd.concat([m1, m2, m3], ignore_index=True)
    df = df.dropna(subset=["a","c","M"]).query("a >= 0 and c > 0 and M > 0").reset_index(drop=True)
    return df


# --- Q-LGBM scoring helpers ------------------------------------------------

def predict_qlgbm_block(fits: dict, X: np.ndarray, prefix: str):
    """Return (q05, q50, q95) in log-space using fits[prefix_q05/q50/q95]."""
    q05 = fits[f"{prefix}_q05"].predict(X)
    q50 = fits[f"{prefix}_q50"].predict(X)
    q95 = fits[f"{prefix}_q95"].predict(X)
    return q05, q50, q95


def quantile_to_moments(log_q05, log_q50, log_q95, sigma_clip=2.0):
    """LogNormal moments from quantile triplet, with sigma clipping for stability."""
    mu_log  = log_q50
    sig_log = (log_q95 - log_q05) / (2.0 * Z95)
    sig_log = np.clip(np.maximum(sig_log, 0.0), 0.0, sigma_clip)
    mean_raw = np.exp(mu_log + 0.5 * sig_log**2)
    var_raw  = (np.exp(sig_log**2) - 1.0) * np.exp(2*mu_log + sig_log**2)
    std_raw  = np.sqrt(np.maximum(var_raw, 0.0))
    q50_raw  = np.exp(mu_log)
    q95_raw  = np.exp(log_q95)
    return mean_raw, std_raw, q50_raw, q95_raw


def b2mb(x): return x / (1024.0 ** 2)


# --- main ------------------------------------------------------------------

def main():
    df = load_all()
    print(f"Loaded {len(df):,} eBPF rows across {df['workflow'].nunique()} workflows")

    with open(HERE/"models_exp2.pkl", "rb") as f:
        M = pickle.load(f)
    pb_models = M["per_bucket"]
    glob = M["global"]
    print(f"Per-bucket models: {len(pb_models)}; global keys: {len(glob)}")

    # Global proxy fitted on the whole pool — used for rows whose bucket
    # has no per-bucket Ridge (small bucket / never trained).
    a_all = np.log(df["a"].values + EPS)
    c_all = np.log(df["c"].values + EPS)
    M_all = np.log(df["M"].values + EPS)
    X_proxy_all = np.column_stack([a_all, c_all, c_all - a_all])
    global_proxy = Ridge(alpha=1.0).fit(X_proxy_all, M_all)
    print(f"Global proxy: intercept={global_proxy.intercept_:.3f}, "
          f"coef={global_proxy.coef_}")

    # --- per-row prediction in vectorised per-bucket batches ---
    out_rows = []
    grouped = df.groupby(["workflow", "process"], sort=False)
    n_per_bucket = n_global_qlgbm = n_global_proxy = 0

    for (wf, proc), grp in grouped:
        idx = grp.index.values
        a_log = np.log(grp["a"].values + EPS)
        c_log = np.log(grp["c"].values + EPS)
        ca_log = c_log - a_log
        X_sizey = a_log.reshape(-1, 1)
        X_joint = np.column_stack([a_log, c_log])
        X_proxy = np.column_stack([a_log, c_log, ca_log])

        key = (wf, proc)
        bucket = pb_models.get(key)

        if bucket is not None and "qlgbm_joint_q50" in bucket["fits"]:
            j05, j50, j95 = predict_qlgbm_block(bucket["fits"], X_joint, "qlgbm_joint")
            s05, s50, s95 = predict_qlgbm_block(bucket["fits"], X_sizey, "qlgbm_sizey")
            n_per_bucket += len(grp)
        else:
            j05, j50, j95 = predict_qlgbm_block(glob, X_joint, "qlgbm_joint")
            s05, s50, s95 = predict_qlgbm_block(glob, X_sizey, "qlgbm_sizey")
            n_global_qlgbm += len(grp)

        mean_j, std_j, q50_j, q95_j = quantile_to_moments(j05, j50, j95)
        mean_s, std_s, q50_s, q95_s = quantile_to_moments(s05, s50, s95)
        safe_j = q95_j
        safe_s = q95_s

        # Proxy
        if bucket is not None and bucket.get("proxy") is not None:
            proxy = bucket["proxy"]
        else:
            proxy = global_proxy
            n_global_proxy += len(grp)
        proxy_log_pred = proxy.predict(X_proxy)
        proxy_pred_b = np.exp(proxy_log_pred)
        coef = proxy.coef_
        intercept = float(proxy.intercept_)

        for i, ridx in enumerate(idx):
            row = df.loc[ridx]
            out_rows.append({
                "workflow": row["workflow"],
                "process":  row["process"],
                "task_hash": row["task_hash"],
                "a_MB": b2mb(row["a"]),
                "c_MB": b2mb(row["c"]),
                "pred_qlgbm_sizey_MB": b2mb(mean_s[i]),
                "std_qlgbm_sizey_MB":  b2mb(std_s[i]),
                "q50_qlgbm_sizey_MB":  b2mb(q50_s[i]),
                "q95_qlgbm_sizey_MB":  b2mb(q95_s[i]),
                "safe_qlgbm_sizey_MB": b2mb(safe_s[i]),
                "pred_qlgbm_joint_MB": b2mb(mean_j[i]),
                "std_qlgbm_joint_MB":  b2mb(std_j[i]),
                "q50_qlgbm_joint_MB":  b2mb(q50_j[i]),
                "q95_qlgbm_joint_MB":  b2mb(q95_j[i]),
                "safe_qlgbm_joint_MB": b2mb(safe_j[i]),
                "proxy_pred_M_MB":     b2mb(proxy_pred_b[i]),
                "proxy_intercept":     intercept,
                "proxy_log_c_coef":    float(coef[1]),
                "proxy_log_a_coef":    float(coef[0]),
                "proxy_log_ca_ratio_coef": float(coef[2]),
                "M_MB":                b2mb(row["M"]),
            })

    out = pd.DataFrame(out_rows)
    out_path = HERE / "predictions_qlgbm_exp2_all.csv"
    out.to_csv(out_path, index=False)

    print(f"\n=== Wrote {out_path}")
    print(f"  rows: {len(out):,}")
    print(f"  per-bucket Q-LGBM: {n_per_bucket:,}  global fallback: {n_global_qlgbm:,}")
    print(f"  global-proxy fallback: {n_global_proxy:,}")

    # Quick sanity: per-bucket MAPE + q95 coverage for both variants
    for tag in ("sizey", "joint"):
        mape  = (out[f"pred_qlgbm_{tag}_MB"].sub(out["M_MB"]).abs() / out["M_MB"]).median() * 100
        cov95 = (out["M_MB"] <= out[f"q95_qlgbm_{tag}_MB"]).mean() * 100
        print(f"  {tag:5s}: median APE {mape:.1f}%   q95 coverage {cov95:.1f}%")


if __name__ == "__main__":
    main()
