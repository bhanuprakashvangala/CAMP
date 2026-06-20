#!/usr/bin/env python3
"""
Train per-task-type models on the full 2062-row dataset.

Per bucket (workflow, process) with >=10 instances:
  - LightGBM Sizey(a)        — trained on ALL rows in bucket (incl. iwd NaN-c rows)
  - NGBoost(a)               — trained on ALL rows
  - TabPFN(a)                — trained on ALL rows (capped at 1000)
  - LightGBM Joint(a,c)      — trained on c-present subset only
  - TabPFN Joint(a,c)        — trained on c-present subset only

Per-method residual std (LOO/k-fold CV) computed on each method's training set,
in log space. Safety cushion = SAFETY_K * residual_std added to log prediction.

Outputs:
  models_unified/per_task_full.pkl   — pickled per-bucket model dict
  models_unified/per_task_full.json  — non-pickle metadata
  prints headline table: GB-hour wastage + failure rate per method
"""
from __future__ import annotations
import json, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor

try:
    from ngboost import NGBRegressor
    from ngboost.distns import LogNormal
    HAS_NGB = True
except ImportError:
    HAS_NGB = False
HAS_TABPFN = False  # disabled: TabPFN phones home to Prior Labs (downloads weights / cloud inference)

REPO        = Path(__file__).parent
OURS_PATH   = REPO / "output" / "joined" / "all_workflows.csv"
TRACE_PATH  = REPO.parent / "trace_methylseq.csv"
MODEL_DIR   = REPO / "models_unified"
MODEL_DIR.mkdir(exist_ok=True)

EPS = 1.0
MIN_INSTANCES = 10
SAFETY_K = 1.5
TABPFN_CAP = 1000  # TabPFN's hard cap on training rows

def mk_lgbm():
    return LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                         min_data_in_leaf=2, verbose=-1)

def mk_ngboost(seed=42):
    if not HAS_NGB: return None
    return NGBRegressor(Dist=LogNormal, n_estimators=200, learning_rate=0.05,
                        verbose=False, random_state=seed)

def mk_tabpfn():
    if not HAS_TABPFN: return None
    return TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)

def cv_folds(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    chunks = np.array_split(idx, min(k, n))
    for i, te in enumerate(chunks):
        tr = np.concatenate([c for j,c in enumerate(chunks) if j != i])
        if len(tr) and len(te):
            yield tr, te

def gb_hour(M_true, M_alloc, runtime_s):
    waste_bytes = np.maximum(M_alloc - M_true, 0)
    return float(np.sum(waste_bytes / (1024**3) * runtime_s / 3600))

def main():
    # ---- 1. Load
    print(f"loading {OURS_PATH.name} + {TRACE_PATH.name}...")
    ours = pd.read_csv(OURS_PATH)
    trace = pd.read_csv(TRACE_PATH)
    on = pd.DataFrame({
        "workflow": ours["workflow"], "process": ours["process"],
        "a": pd.to_numeric(ours["a_bytes"], errors="coerce"),
        "c": pd.to_numeric(ours["c_bytes"], errors="coerce"),
        "M": pd.to_numeric(ours["M_peak_rss_bytes"], errors="coerce"),
        "runtime_sec": pd.to_numeric(ours["runtime_seconds"], errors="coerce"),
    }).dropna(subset=["a","M"]).query("M>0 and a>=0").reset_index(drop=True)
    tn = pd.DataFrame({
        "workflow":"methylseq_naga", "process": trace["process"],
        "a": pd.to_numeric(trace["read_bytes"], errors="coerce"),
        "c": pd.to_numeric(trace["rchar"], errors="coerce"),
        "M": pd.to_numeric(trace["peak_rss"], errors="coerce"),
        "runtime_sec": pd.to_numeric(trace["realtime"], errors="coerce")/1000.0,
    }).dropna(subset=["a","c","M"]).query("M>0 and c>0 and a>=0").reset_index(drop=True)
    df = pd.concat([on, tn], ignore_index=True)
    df["runtime_sec"] = df["runtime_sec"].fillna(60.0)  # conservative default
    print(f"  ours={len(on)}  trace_methylseq={len(tn)}  combined={len(df)}")
    print(f"  with c: {df['c'].notna().sum()} / {len(df)}")

    # ---- 2. Bucket
    bucket_keys = (df.groupby(["workflow","process"]).size()
                   .reset_index(name="n").query(f"n>={MIN_INSTANCES}"))
    print(f"  buckets >= {MIN_INSTANCES}: {len(bucket_keys)} (rows in trainable buckets: {bucket_keys.n.sum()})")

    # ---- 3. Train per bucket
    rows = []
    to_pickle = {}
    t0 = time.time()
    for _, br in bucket_keys.iterrows():
        wf, proc, n = br.workflow, br.process, int(br.n)
        grp = df[(df.workflow==wf) & (df.process==proc)].reset_index(drop=True)
        a_log = np.log(grp["a"].values + EPS)
        M_log = np.log(grp["M"].values + EPS)
        M = grp["M"].values
        rt = grp["runtime_sec"].values
        c_mask = grp["c"].notna().values
        n_c = int(c_mask.sum())

        # pre-allocate prediction arrays (NaN where N/A)
        p_lgbm_sizey = np.full(n, np.nan)
        p_ngb_a      = np.full(n, np.nan)
        p_ngb_a_sig  = np.full(n, np.nan)
        p_tabpfn_a   = np.full(n, np.nan)
        p_lgbm_joint = np.full(n, np.nan)
        p_tabpfn_joint = np.full(n, np.nan)

        # ---- a-only models: trained on ALL rows in bucket
        for tr, te in cv_folds(n):
            Xtr, Xte = a_log[tr].reshape(-1,1), a_log[te].reshape(-1,1)
            ytr = M_log[tr]
            m_lgbm = mk_lgbm().fit(Xtr, ytr)
            p_lgbm_sizey[te] = m_lgbm.predict(Xte)
            if HAS_NGB:
                try:
                    m_ngb = mk_ngboost().fit(Xtr, np.exp(ytr))
                    dist = m_ngb.pred_dist(Xte)
                    p_ngb_a[te]     = np.log(np.maximum(1.0, dist.mean()))
                    p_ngb_a_sig[te] = np.maximum(np.std(np.log(dist.sample(50)), axis=0), 1e-3)
                except Exception:
                    p_ngb_a[te] = p_lgbm_sizey[te]
                    p_ngb_a_sig[te] = 0.5
            if HAS_TABPFN and len(tr) <= TABPFN_CAP:
                try:
                    t = mk_tabpfn(); t.fit(Xtr, ytr)
                    p_tabpfn_a[te] = t.predict(Xte)
                except Exception:
                    p_tabpfn_a[te] = p_lgbm_sizey[te]
            else:
                p_tabpfn_a[te] = p_lgbm_sizey[te]

        # ---- a+c joint: trained on c-present subset of bucket
        if n_c >= MIN_INSTANCES:
            idx_c = np.flatnonzero(c_mask)
            a_log_c = a_log[idx_c]
            c_log_c = np.log(grp["c"].values[idx_c] + EPS)
            M_log_c = M_log[idx_c]
            for tr_loc, te_loc in cv_folds(n_c):
                Xtr = np.column_stack([a_log_c[tr_loc], c_log_c[tr_loc]])
                Xte = np.column_stack([a_log_c[te_loc], c_log_c[te_loc]])
                ytr = M_log_c[tr_loc]
                m_lgbm = mk_lgbm().fit(Xtr, ytr)
                p_lgbm_joint[idx_c[te_loc]] = m_lgbm.predict(Xte)
                if HAS_TABPFN and len(tr_loc) <= TABPFN_CAP:
                    try:
                        t = mk_tabpfn(); t.fit(Xtr, ytr)
                        p_tabpfn_joint[idx_c[te_loc]] = t.predict(Xte)
                    except Exception:
                        p_tabpfn_joint[idx_c[te_loc]] = m_lgbm.predict(Xte)
                else:
                    p_tabpfn_joint[idx_c[te_loc]] = m_lgbm.predict(Xte)

        # ---- residual std per method (in log space, on the method's eval rows)
        def resid_std(pred):
            mask = ~np.isnan(pred)
            if mask.sum() < 2: return 0.5
            return float(np.std(M_log[mask] - pred[mask]))
        rs_sizey  = resid_std(p_lgbm_sizey)
        rs_ngb    = resid_std(p_ngb_a)
        rs_tab_a  = resid_std(p_tabpfn_a)
        rs_joint  = resid_std(p_lgbm_joint)
        rs_tab_j  = resid_std(p_tabpfn_joint)

        # ---- safe-allocation = exp(pred + K*resid_std)  -> wastage + failure
        def metrics(pred, rs):
            mask = ~np.isnan(pred)
            if mask.sum() == 0: return None
            safe = np.exp(pred[mask] + SAFETY_K * rs)
            return dict(n=int(mask.sum()),
                        wastage_GBh=gb_hour(M[mask], safe, rt[mask]),
                        failures=int((safe < M[mask]).sum()))
        rows.append(dict(workflow=wf, process=proc, n=n, n_c=n_c,
                         lgbm_sizey=metrics(p_lgbm_sizey, rs_sizey),
                         ngb_a=metrics(p_ngb_a, rs_ngb),
                         tabpfn_a=metrics(p_tabpfn_a, rs_tab_a),
                         lgbm_joint=metrics(p_lgbm_joint, rs_joint),
                         tabpfn_joint=metrics(p_tabpfn_joint, rs_tab_j)))

        # ---- refit on full bucket for production save
        sizey_full = mk_lgbm().fit(a_log.reshape(-1,1), M_log)
        joint_full = None
        if n_c >= MIN_INSTANCES:
            idx_c = np.flatnonzero(c_mask)
            joint_full = mk_lgbm().fit(
                np.column_stack([a_log[idx_c], np.log(grp["c"].values[idx_c]+EPS)]),
                M_log[idx_c])
        to_pickle[f"{wf}::{proc}"] = {
            "sizey": sizey_full, "joint": joint_full,
            "n": n, "n_c": n_c,
            "resid_sizey": rs_sizey, "resid_joint": rs_joint,
            "safety_k": SAFETY_K,
        }

    elapsed = time.time() - t0
    print(f"  trained {len(rows)} buckets in {elapsed:.1f}s")

    # ---- 4. Headline aggregate (sum across buckets)
    print("\n=== HEADLINE TABLE — wastage + failures aggregated across all buckets ===")
    print(f"{'method':<18} {'buckets':>8} {'rows':>8} {'wastage_GBh':>14} {'failures':>10}")
    print("-" * 62)
    for col in ["lgbm_sizey", "ngb_a", "tabpfn_a", "lgbm_joint", "tabpfn_joint"]:
        nb = sum(1 for r in rows if r[col])
        nr = sum(r[col]["n"] for r in rows if r[col])
        wg = sum(r[col]["wastage_GBh"] for r in rows if r[col])
        fl = sum(r[col]["failures"] for r in rows if r[col])
        print(f"{col:<18} {nb:>8} {nr:>8} {wg:>14.1f} {fl:>10}")

    # ---- 5. Save
    PKL  = MODEL_DIR / "per_task_full.pkl"
    META = MODEL_DIR / "per_task_full.json"
    with open(PKL, "wb") as f: pickle.dump(to_pickle, f)
    with open(META, "w") as f:
        json.dump({k: {kk:vv for kk,vv in v.items() if kk not in ("sizey","joint")}
                   for k,v in to_pickle.items()}, f, indent=2)
    print(f"\nwrote {PKL} ({PKL.stat().st_size:,} bytes)")
    print(f"wrote {META}")

    # ---- 6. Per-bucket summary (top 5 by row count)
    print("\n=== TOP 10 BUCKETS BY ROW COUNT ===")
    print(f"{'workflow':<18} {'process':<60} {'n':>5} {'n_c':>5}")
    print("-" * 92)
    for r in sorted(rows, key=lambda x: -x["n"])[:10]:
        print(f"{r['workflow']:<18} {r['process'][:58]:<60} {r['n']:>5} {r['n_c']:>5}")

if __name__ == "__main__":
    main()
