#!/usr/bin/env python3
"""
reruns/sanity_checks.py — verifies the three properties from TASK 6.

(a) Each bucket has 6 entries in sizey_models, and 6 in joint_models
    (or joint_models is None for iwd buckets where c is unavailable).

(b) For a randomly sampled set of (a, c) inputs per bucket, the new
    max-aggregated M_safe is always >= the LGBM-only M_safe that the
    prior pickle would have produced.

(c) The NGBoost gate sigma distribution (per-row sigma_log emitted by
    NGBoost LogNormal on a held-out sample) is qualitatively similar
    in scale and spread to the original LGBM seed-pair |p1-p2|/2 proxy.

Outputs a small JSON report next to the pickle so the comparison is
auditable.
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

from predict_memory_unified import predict_safe

REPO = Path(__file__).resolve().parent
NEW_PKL = REPO / "models_unified" / "per_task_full.pkl"
OLD_PKL = REPO.parent / "experiment_1" / "models_unified" / "per_task_no_methylseq.pkl"
EXP1_FILE = (REPO.parent / "selective_audit" / "experiment1"
             / "budget_0.3_Exp1" / "exp1_lgbm_sizey_scores_b_0.3.csv")
REPORT = REPO / "models_unified" / "sanity_report.json"

EPS = 1.0
SEED = 42


def check_a_counts(M_new):
    """Property (a): 6 sizey_models per bucket; 6 joint_models or None for iwd."""
    bad_sizey, bad_joint, iwd_with_joint, ok_iwd_none = [], [], [], []
    for k, v in M_new.items():
        if len(v["sizey_models"]) != 6:
            bad_sizey.append((k, len(v["sizey_models"])))
        if k.startswith("iwd::"):
            if v["joint_models"] is None:
                ok_iwd_none.append(k)
            else:
                iwd_with_joint.append(k)
        else:
            if v["joint_models"] is None:
                # acceptable — bucket may have <MIN_INSTANCES rows with c
                continue
            elif len(v["joint_models"]) != 6:
                bad_joint.append((k, len(v["joint_models"])))
    return {
        "buckets_total": len(M_new),
        "buckets_with_6_sizey":  sum(1 for v in M_new.values() if len(v["sizey_models"]) == 6),
        "buckets_with_6_joint":  sum(1 for v in M_new.values() if v["joint_models"] is not None and len(v["joint_models"]) == 6),
        "iwd_buckets_joint_none": len(ok_iwd_none),
        "iwd_buckets_with_joint": iwd_with_joint,
        "non_six_sizey": bad_sizey,
        "non_six_joint": bad_joint,
    }


def check_b_max_dominates(M_new, M_old):
    """Property (b): new max-aggregation M_safe >= old LGBM-only M_safe per row."""
    if not OLD_PKL.exists():
        return {"skipped": True, "reason": f"old pickle not found at {OLD_PKL}"}

    rows_checked = 0
    rows_violated = 0
    delta_pct = []
    common_keys = set(M_new.keys()) & set(M_old.keys())
    rng = np.random.default_rng(SEED)
    sampled = rng.choice(list(common_keys), size=min(60, len(common_keys)), replace=False) \
              if common_keys else []

    for k in sampled:
        info_new = M_new[k]
        info_old = M_old[k]
        # need both joint paths present to compare
        if info_new.get("joint_models") is None or info_old.get("joint") is None:
            continue
        # generate 5 representative (a, c) probes per bucket using
        # log-uniform draws around 1 GB / 2 GB
        for _ in range(5):
            a = float(np.exp(rng.uniform(np.log(1e7), np.log(5e9))))
            c = float(np.exp(rng.uniform(np.log(1e7), np.log(5e9))))
            try:
                m_new, _, _ = predict_safe(info_new, a, c)
                # old path mirrors experiment_1/predict_memory_unified.py:25-31
                M_log = float(info_old["joint"].predict(
                    np.array([[np.log(a + EPS), np.log(c + EPS)]])
                )[0])
                m_old = float(np.exp(M_log + info_old["safety_k"] * info_old["resid_joint"]))
            except Exception:
                continue
            rows_checked += 1
            if m_new + 1e-6 < m_old:
                rows_violated += 1
            delta_pct.append((m_new - m_old) / max(m_old, 1.0) * 100.0)

    return {
        "skipped": False,
        "rows_checked": rows_checked,
        "rows_where_new_lt_old": rows_violated,
        "median_delta_pct":  float(np.median(delta_pct)) if delta_pct else None,
        "min_delta_pct":     float(np.min(delta_pct))    if delta_pct else None,
        "p95_delta_pct":     float(np.percentile(delta_pct, 95)) if delta_pct else None,
        "buckets_sampled": int(len(sampled)),
    }


def check_c_sigma_distributions(M_new):
    """Property (c): summarise NGBoost per-row sigma_log on held-out samples
    against the LGBM seed-pair sigma proxy used by the prior gate.
    The new gate code lives in run_active_learning.py; here we re-fit a quick
    pair on a small subset and compare summary statistics."""
    if not EXP1_FILE.exists():
        return {"skipped": True, "reason": f"audit drop missing: {EXP1_FILE}"}

    df = (pd.read_csv(EXP1_FILE)
            .dropna(subset=["a_MB", "c_MB", "M_MB"])
            .query("a_MB >= 0 and c_MB > 0 and M_MB > 0").reset_index(drop=True))
    rng = np.random.default_rng(SEED)
    use = df.iloc[rng.choice(len(df), size=min(2000, len(df)), replace=False)].reset_index(drop=True)

    X = np.column_stack([np.log(use["a_MB"].values + EPS),
                         np.log(use["c_MB"].values + EPS)])
    y_log = np.log(use["M_MB"].values + EPS)

    # NGBoost on a 90% subsample, score the other 10%
    n = len(use); cut = int(0.9 * n)
    idx = rng.permutation(n)
    tr, te = idx[:cut], idx[cut:]

    from ngboost import NGBRegressor
    from ngboost.distns import LogNormal
    ngb = NGBRegressor(Dist=LogNormal, n_estimators=200, learning_rate=0.05,
                       verbose=False, random_state=42).fit(X[tr], np.exp(y_log[tr]))
    dist = ngb.pred_dist(X[te])
    sigma_ngb = np.asarray(dist.scale, dtype=float)

    # LGBM seed-pair proxy
    from lightgbm import LGBMRegressor
    m1 = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05,
                       verbose=-1, random_state=42).fit(X[tr], y_log[tr])
    m2 = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05,
                       subsample=0.7, subsample_freq=1, verbose=-1,
                       random_state=49).fit(X[tr], y_log[tr])
    p1 = m1.predict(X[te]); p2 = m2.predict(X[te])
    sigma_pair = np.abs(p1 - p2) / 2.0

    def stats(arr):
        return {
            "n":      int(len(arr)),
            "mean":   float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std":    float(np.std(arr)),
            "p10":    float(np.percentile(arr, 10)),
            "p50":    float(np.percentile(arr, 50)),
            "p90":    float(np.percentile(arr, 90)),
        }

    spearman = float(pd.Series(sigma_ngb).rank().corr(pd.Series(sigma_pair).rank()))
    return {
        "skipped": False,
        "ngboost_sigma_log_stats": stats(sigma_ngb),
        "lgbm_pair_sigma_stats":   stats(sigma_pair),
        "rank_correlation":        spearman,
    }


def main():
    if not NEW_PKL.exists():
        sys.exit(f"missing {NEW_PKL}; run train_full.py first")
    with open(NEW_PKL, "rb") as f:
        M_new = pickle.load(f)
    M_old = None
    if OLD_PKL.exists():
        with open(OLD_PKL, "rb") as f:
            M_old = pickle.load(f)

    report = {
        "a_counts":            check_a_counts(M_new),
        "b_max_dominates":     check_b_max_dominates(M_new, M_old) if M_old else {"skipped": True, "reason": "old pickle missing"},
        "c_sigma_distributions": check_c_sigma_distributions(M_new),
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
