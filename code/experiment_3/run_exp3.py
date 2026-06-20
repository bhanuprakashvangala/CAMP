#!/usr/bin/env python3
"""
Experiment 3 — active-learning retraining + held-out evaluation.

Inputs: Naga's selective-audit drops at B in {0.1, 0.3} for both experiments.

For each (experiment, B) we evaluate four deploy strategies on a fixed
20% held-out partition (per-bucket, seed=42):
  - Sizey-only      (B = 0,  no audit cost, joint never used)
  - Random @ B      (random subset of training pool gets audited)
  - Gate   @ B      (Naga's combined-gate audit_flag drives selection)
  - Oracle          (B = 1, every training row audited, joint used at deploy)

For each strategy we report two retraining modes:
  - Naive joint     (joint trained on audited training rows, sample weights = 1)
  - IPW   joint     (sample weights = 1/p_t with p_t calibrated so E[Σp]=B·N)

At deploy, the held-out task is routed to the joint predictor if the gate
decides to audit at deployment (uses Naga's audit_flag on the held-out side);
otherwise it falls back to the Sizey predictor.

Metrics on held-out:
  MAPE_med   median of |pred - M| / M
  R2_log     1 - SS_res / SS_tot in log space
  Cov2x      fraction with 0.5 <= pred/M <= 2
  Wastage_MB sum of max(safe - M, 0) over held-out rows
  OOM_count  count of rows with safe < M
"""

from __future__ import annotations
from pathlib import Path
import warnings, json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "selective_audit"
OUT   = Path(__file__).resolve().parent
EPS   = 1.0
SEED  = 42
Z95   = 1.6448536269514722
K_SAFETY = 1.5         # log-space safety margin coefficient
RHO_EBPF = 0.10        # 10% audit overhead (per §1 lower bound)

EXP1_FILES = {
    0.1: AUDIT / "experiment1" / "budge_0.1_Exp1"  / "exp1_lgbm_sizey_scores_b_0.1.csv",
    0.3: AUDIT / "experiment1" / "budget_0.3_Exp1" / "exp1_lgbm_sizey_scores_b_0.3.csv",
}
EXP2_FILES = {
    0.1: AUDIT / "experiment1" / "experiment2" / "budget_0.1_Exp2" / "exp2_qlgbm_sizey_scores_b_0.1.csv",
    0.3: AUDIT / "experiment1" / "experiment2" / "budget_0.1_Exp2" / "budget_0.3_Exp2" / "exp2_qlgbm_sizey_scores_b_0.3.csv",
}


# --- helpers ---------------------------------------------------------------

def split_80_20(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `partition` column ('train' / 'test') with per-bucket 80/20 split."""
    rng = np.random.RandomState(SEED)
    df = df.copy()
    df["partition"] = "train"
    for (wf, proc), grp in df.groupby(["workflow", "process"], sort=False):
        idx = grp.index.values.copy()
        rng.shuffle(idx)
        cut = int(len(idx) * 0.8)
        df.loc[idx[cut:], "partition"] = "test"
    return df


def calibrate_ipw(scores: np.ndarray, audit_flag: np.ndarray, B: float):
    """
    Calibrate a logistic mapping  p(score) = 1 / (1 + exp(-(score - tau) / T))
    such that mean(p) ~= B.  Returns (tau, T) and a vectorised p().
    """
    s = np.asarray(scores, dtype=float)
    target = float(B)
    # tau = score quantile that puts B-fraction above (matches Naga's deterministic top-B)
    tau = float(np.quantile(s, 1.0 - target))
    # search T so that mean(sigmoid((s - tau)/T)) ≈ B
    best_T, best_err = 0.05, 1e9
    for T in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]:
        p = 1.0 / (1.0 + np.exp(-(s - tau) / T))
        err = abs(p.mean() - target)
        if err < best_err:
            best_err, best_T = err, T
    def p_fn(x):
        x = np.asarray(x, dtype=float)
        return 1.0 / (1.0 + np.exp(-(x - tau) / best_T))
    return tau, best_T, p_fn


def fit_lgbm_sizey(df_tr, weights=None):
    X = np.log(df_tr["a_MB"].values + EPS).reshape(-1, 1)
    y = np.log(df_tr["M_MB"].values + EPS)
    m = LGBMRegressor(n_estimators=200, max_depth=-1, num_leaves=31,
                       learning_rate=0.05, verbose=-1, random_state=SEED)
    m.fit(X, y, sample_weight=weights)
    return m


def fit_lgbm_joint(df_tr, weights=None):
    X = np.column_stack([np.log(df_tr["a_MB"].values + EPS),
                          np.log(df_tr["c_MB"].values + EPS)])
    y = np.log(df_tr["M_MB"].values + EPS)
    m = LGBMRegressor(n_estimators=200, max_depth=-1, num_leaves=31,
                       learning_rate=0.05, verbose=-1, random_state=SEED)
    m.fit(X, y, sample_weight=weights)
    return m


def fit_qlgbm_quantile(df_tr, q, joint=True, weights=None):
    if joint:
        X = np.column_stack([np.log(df_tr["a_MB"].values + EPS),
                              np.log(df_tr["c_MB"].values + EPS)])
    else:
        X = np.log(df_tr["a_MB"].values + EPS).reshape(-1, 1)
    y = np.log(df_tr["M_MB"].values + EPS)
    m = LGBMRegressor(objective="quantile", alpha=q,
                       n_estimators=200, max_depth=-1, num_leaves=31,
                       learning_rate=0.05, verbose=-1, random_state=SEED)
    m.fit(X, y, sample_weight=weights)
    return m


def predict_lgbm(model, df, joint=True):
    if joint:
        X = np.column_stack([np.log(df["a_MB"].values + EPS),
                              np.log(df["c_MB"].values + EPS)])
    else:
        X = np.log(df["a_MB"].values + EPS).reshape(-1, 1)
    return np.exp(model.predict(X))


def safe_from_residuals(model, df_tr, joint, K=K_SAFETY):
    """Compute log-residual std on training set and return safety multiplier."""
    if joint:
        X = np.column_stack([np.log(df_tr["a_MB"].values + EPS),
                              np.log(df_tr["c_MB"].values + EPS)])
    else:
        X = np.log(df_tr["a_MB"].values + EPS).reshape(-1, 1)
    y = np.log(df_tr["M_MB"].values + EPS)
    pred = model.predict(X)
    resid = y - pred
    sigma = float(resid.std()) if len(resid) > 1 else 0.0
    return sigma  # use as: safe = pred * exp(K * sigma)


def metrics(pred_mb, safe_mb, M_mb, runtime_sec=None) -> dict:
    pred = np.asarray(pred_mb, dtype=float)
    safe = np.asarray(safe_mb, dtype=float)
    M    = np.asarray(M_mb, dtype=float)
    mask = (M > 0) & np.isfinite(pred) & np.isfinite(M)
    pred, safe, M = pred[mask], safe[mask], M[mask]
    if runtime_sec is not None:
        rt = np.asarray(runtime_sec, dtype=float)[mask]
    else:
        rt = np.ones_like(M)
    ape = np.abs(pred - M) / M
    mape = float(np.median(ape) * 100)
    log_y = np.log(M + EPS); log_p = np.log(np.clip(pred, EPS, None))
    ss_res = float(np.sum((log_y - log_p) ** 2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    cov2x = float(((pred / M >= 0.5) & (pred / M <= 2.0)).mean() * 100)
    over = np.maximum(safe - M, 0.0)
    wastage_mb = float(over.sum())
    wastage_gbh = float((over * rt / 3600.0 / 1024.0).sum())
    oom = int((safe < M).sum())
    return {
        "MAPE_med": mape, "R2_log": r2, "Cov2x": cov2x,
        "Wastage_MB": wastage_mb, "Wastage_GBh": wastage_gbh,
        "OOM_count": oom, "n_eval": int(len(M)),
    }


# --- Exp 1 pipeline --------------------------------------------------------

def run_exp1(B: float, df_full: pd.DataFrame) -> list[dict]:
    """Train Sizey + Joint variants under each strategy at budget B; eval on held-out."""
    df = split_80_20(df_full)
    # IPW probability is computed on the full frame so train/test inherit it
    tau, T, p_fn = calibrate_ipw(df["final_score"].values,
                                   (df["audit_flag"] == "Audit").astype(int).values, B)
    df["audit_p"] = p_fn(df["final_score"].values)
    tr_all = df.query("partition == 'train'").copy()
    te     = df.query("partition == 'test'").copy()
    print(f"\n=== Exp 1 @ B={B} ===   train={len(tr_all):,}  test={len(te):,}")
    print(f"  IPW calibration: tau={tau:.3f}  T={T:.3f}  mean(p)={df['audit_p'].mean():.3f} (target {B})")

    # Base Sizey on full training pool — used as the no-audit fallback
    sizey = fit_lgbm_sizey(tr_all)
    sigma_sizey = safe_from_residuals(sizey, tr_all, joint=False)

    # Oracle Joint on full training pool — represents B=1 deploy
    oracle = fit_lgbm_joint(tr_all)
    sigma_oracle = safe_from_residuals(oracle, tr_all, joint=True)

    # ---- Random @ B subset (training side) ----
    rng = np.random.RandomState(SEED + int(B*1000))
    n_audit_rand = int(round(B * len(tr_all)))
    rand_audit_idx = rng.choice(tr_all.index.values, size=n_audit_rand, replace=False)
    tr_rand = tr_all.loc[rand_audit_idx].copy()

    # ---- Gate @ B subset (training side) — Naga's audit_flag restricted to training rows ----
    tr_gate = tr_all[tr_all["audit_flag"] == "Audit"].copy()
    print(f"  Random pool: {len(tr_rand)}   Gate pool: {len(tr_gate)}   "
          f"(target {n_audit_rand})")

    # Train joint on each audit subset (naive; n>=10)
    def maybe_fit_joint(sub_df, label):
        if len(sub_df) < 20:
            return None, 0.0
        m = fit_lgbm_joint(sub_df)
        s = safe_from_residuals(m, sub_df, joint=True)
        return m, s

    joint_rand_naive, sig_rand_naive = maybe_fit_joint(tr_rand, "rand_naive")
    joint_gate_naive, sig_gate_naive = maybe_fit_joint(tr_gate, "gate_naive")

    # IPW retraining: weights = 1 / p_t for audited rows (no rescaling — sample-weight only)
    w_rand = 1.0 / np.clip(B, 1e-3, None) * np.ones(len(tr_rand))  # uniform random selection at B
    w_gate = 1.0 / np.clip(tr_gate["audit_p"].values, 1e-3, None)
    if len(tr_rand) >= 20:
        joint_rand_ipw = fit_lgbm_joint(tr_rand, weights=w_rand)
        sig_rand_ipw   = safe_from_residuals(joint_rand_ipw, tr_rand, joint=True)
    else:
        joint_rand_ipw, sig_rand_ipw = None, 0.0
    if len(tr_gate) >= 20:
        joint_gate_ipw = fit_lgbm_joint(tr_gate, weights=w_gate)
        sig_gate_ipw   = safe_from_residuals(joint_gate_ipw, tr_gate, joint=True)
    else:
        joint_gate_ipw, sig_gate_ipw = None, 0.0

    # --- Held-out evaluation ---
    pred_sizey = predict_lgbm(sizey, te, joint=False)
    safe_sizey = pred_sizey * np.exp(K_SAFETY * sigma_sizey)
    pred_oracle = predict_lgbm(oracle, te, joint=True)
    safe_oracle = pred_oracle * np.exp(K_SAFETY * sigma_oracle)

    # held-out audit decisions: use Naga's audit_flag on test rows (gate strategy)
    te_gate_audit = (te["audit_flag"].values == "Audit")
    # random strategy: independent random B-fraction of held-out
    te_rand_audit = rng.random(len(te)) < B

    def hybrid(joint_model, sigma_joint, te_audit_mask):
        pred = pred_sizey.copy()
        safe = safe_sizey.copy()
        if joint_model is not None and te_audit_mask.any():
            p_j = predict_lgbm(joint_model, te[te_audit_mask], joint=True)
            s_j = p_j * np.exp(K_SAFETY * sigma_joint)
            pred[te_audit_mask] = p_j
            safe[te_audit_mask] = s_j
        return pred, safe

    rows = []
    rows.append({"experiment": "exp1", "B": 0.0, "strategy": "sizey",   "retrain": "n/a",
                  "n_audit_train": 0, **metrics(pred_sizey, safe_sizey, te["M_MB"].values)})
    rows.append({"experiment": "exp1", "B": 1.0, "strategy": "oracle",  "retrain": "n/a",
                  "n_audit_train": len(tr_all), **metrics(pred_oracle, safe_oracle, te["M_MB"].values)})

    for label, model_naive, sig_naive, model_ipw, sig_ipw, te_mask, n_aud in [
        ("random", joint_rand_naive, sig_rand_naive, joint_rand_ipw, sig_rand_ipw, te_rand_audit, len(tr_rand)),
        ("gate",   joint_gate_naive, sig_gate_naive, joint_gate_ipw, sig_gate_ipw, te_gate_audit, len(tr_gate)),
    ]:
        if model_naive is not None:
            p, s = hybrid(model_naive, sig_naive, te_mask)
            rows.append({"experiment": "exp1", "B": B, "strategy": label, "retrain": "naive",
                          "n_audit_train": n_aud, **metrics(p, s, te["M_MB"].values)})
        if model_ipw is not None:
            p, s = hybrid(model_ipw, sig_ipw, te_mask)
            rows.append({"experiment": "exp1", "B": B, "strategy": label, "retrain": "ipw",
                          "n_audit_train": n_aud, **metrics(p, s, te["M_MB"].values)})
    return rows


# --- Exp 2 pipeline --------------------------------------------------------

def run_exp2(B: float, df_full: pd.DataFrame) -> list[dict]:
    df = split_80_20(df_full)
    tau, T, p_fn = calibrate_ipw(df["final_score"].values,
                                   (df["audit_flag"] == "Audit").astype(int).values, B)
    df["audit_p"] = p_fn(df["final_score"].values)
    tr_all = df.query("partition == 'train'").copy()
    te     = df.query("partition == 'test'").copy()
    print(f"\n=== Exp 2 @ B={B} ===   train={len(tr_all):,}  test={len(te):,}")
    print(f"  IPW calibration: tau={tau:.3f}  T={T:.3f}  mean(p)={df['audit_p'].mean():.3f} (target {B})")

    # Sizey Q-LGBM trio (q05/q50/q95) on full training pool
    sizey50 = fit_qlgbm_quantile(tr_all, 0.50, joint=False)
    sizey95 = fit_qlgbm_quantile(tr_all, 0.95, joint=False)
    # Oracle Joint
    oracle50 = fit_qlgbm_quantile(tr_all, 0.50, joint=True)
    oracle95 = fit_qlgbm_quantile(tr_all, 0.95, joint=True)

    rng = np.random.RandomState(SEED + int(B*1000))
    n_audit_rand = int(round(B * len(tr_all)))
    rand_audit_idx = rng.choice(tr_all.index.values, size=n_audit_rand, replace=False)
    tr_rand = tr_all.loc[rand_audit_idx].copy()
    tr_gate = tr_all[tr_all["audit_flag"] == "Audit"].copy()
    print(f"  Random pool: {len(tr_rand)}   Gate pool: {len(tr_gate)}   "
          f"(target {n_audit_rand})")

    def fit_joint_pair(sub, weights=None):
        if len(sub) < 20: return None, None
        return (fit_qlgbm_quantile(sub, 0.50, joint=True, weights=weights),
                fit_qlgbm_quantile(sub, 0.95, joint=True, weights=weights))

    rand_naive = fit_joint_pair(tr_rand)
    gate_naive = fit_joint_pair(tr_gate)
    w_rand = 1.0 / np.clip(B, 1e-3, None) * np.ones(len(tr_rand))
    w_gate = 1.0 / np.clip(tr_gate["audit_p"].values, 1e-3, None)
    rand_ipw = fit_joint_pair(tr_rand, weights=w_rand)
    gate_ipw = fit_joint_pair(tr_gate, weights=w_gate)

    # Held-out predictions
    pred_sizey = np.exp(sizey50.predict(np.log(te["a_MB"].values + EPS).reshape(-1, 1)))
    safe_sizey = np.exp(sizey95.predict(np.log(te["a_MB"].values + EPS).reshape(-1, 1)))
    pred_oracle = np.exp(oracle50.predict(np.column_stack(
        [np.log(te["a_MB"].values + EPS), np.log(te["c_MB"].values + EPS)])))
    safe_oracle = np.exp(oracle95.predict(np.column_stack(
        [np.log(te["a_MB"].values + EPS), np.log(te["c_MB"].values + EPS)])))

    te_gate_audit = (te["audit_flag"].values == "Audit")
    te_rand_audit = rng.random(len(te)) < B

    def hybrid_q(pair, te_mask):
        pred = pred_sizey.copy()
        safe = safe_sizey.copy()
        if pair is None or pair[0] is None or not te_mask.any():
            return pred, safe
        m50, m95 = pair
        Xj = np.column_stack([np.log(te[te_mask]["a_MB"].values + EPS),
                                np.log(te[te_mask]["c_MB"].values + EPS)])
        pred[te_mask] = np.exp(m50.predict(Xj))
        safe[te_mask] = np.exp(m95.predict(Xj))
        return pred, safe

    rows = []
    rows.append({"experiment": "exp2", "B": 0.0, "strategy": "sizey",  "retrain": "n/a",
                  "n_audit_train": 0, **metrics(pred_sizey, safe_sizey, te["M_MB"].values)})
    rows.append({"experiment": "exp2", "B": 1.0, "strategy": "oracle", "retrain": "n/a",
                  "n_audit_train": len(tr_all), **metrics(pred_oracle, safe_oracle, te["M_MB"].values)})

    for label, naive, ipw, te_mask, n_aud in [
        ("random", rand_naive, rand_ipw, te_rand_audit, len(tr_rand)),
        ("gate",   gate_naive, gate_ipw, te_gate_audit, len(tr_gate)),
    ]:
        if naive[0] is not None:
            p, s = hybrid_q(naive, te_mask)
            rows.append({"experiment": "exp2", "B": B, "strategy": label, "retrain": "naive",
                          "n_audit_train": n_aud, **metrics(p, s, te["M_MB"].values)})
        if ipw[0] is not None:
            p, s = hybrid_q(ipw, te_mask)
            rows.append({"experiment": "exp2", "B": B, "strategy": label, "retrain": "ipw",
                          "n_audit_train": n_aud, **metrics(p, s, te["M_MB"].values)})
    return rows


# --- main ------------------------------------------------------------------

def main():
    all_rows = []

    # Exp 1: load each B file (same population, just different audit_flag column)
    print("Loading Exp 1 audit data...")
    for B, fp in EXP1_FILES.items():
        d = pd.read_csv(fp)
        d = d.dropna(subset=["a_MB", "c_MB", "M_MB"]).query("a_MB >= 0 and c_MB > 0 and M_MB > 0").reset_index(drop=True)
        all_rows.extend(run_exp1(B, d))

    # Exp 2
    print("\nLoading Exp 2 audit data...")
    for B, fp in EXP2_FILES.items():
        d = pd.read_csv(fp)
        d = d.dropna(subset=["a_MB", "c_MB", "M_MB"]).query("a_MB >= 0 and c_MB > 0 and M_MB > 0").reset_index(drop=True)
        all_rows.extend(run_exp2(B, d))

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT / "results_exp3.csv", index=False)
    print("\n\n========= results_exp3.csv =========")
    print(out.to_string(index=False))
    print(f"\nWrote {OUT/'results_exp3.csv'}")

if __name__ == "__main__":
    main()
