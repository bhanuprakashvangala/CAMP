#!/usr/bin/env python3
"""
reruns/train_full.py — paper-faithful zoo + max-aggregation pickle.

Per bucket b = (workflow, process) with >= MIN_INSTANCES rows:
  For each feature view v in {Sizey, Joint}:
    For each model class i in {LR, kNN, MLP, RF, LightGBM, NGBoost}:
      - run 5-fold CV on the training partition (full bucket here, since we
        deploy the bucket dict; the held-out evaluation lives elsewhere).
      - sigma_{b,i,v} = std( log M - log_pred ) over CV-held-out predictions.
      - refit on the entire bucket and store (name_i, model_i, sigma_{b,i,v}).

Output bucket_dict[bucket_key] = {
    "sizey_models": [(name_i, model_i, sigma_i), ...],   # always populated
    "joint_models": [(name_i, model_i, sigma_i), ...],   # None if c missing
    "n":            int,
    "n_c":          int,
    "safety_k":     1.5,
}

iwd buckets have c missing for every row -> joint_models = None.
"""
from __future__ import annotations
import json, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model    import LinearRegression
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.neural_network  import MLPRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from lightgbm                import LGBMRegressor
from ngboost                 import NGBRegressor
from ngboost.distns          import LogNormal

from _zoo import NGBLogMu


# -------------------- paths / constants --------------------
REPO        = Path(__file__).resolve().parent
EXP1_DIR    = REPO.parent / "experiment_1"
OURS_PATH   = EXP1_DIR / "joined" / "all_workflows.csv"
TRACE_PATH  = EXP1_DIR / "trace_methylseq.csv"
MODEL_DIR   = REPO / "models_unified"
MODEL_DIR.mkdir(exist_ok=True)

EPS           = 1.0
MIN_INSTANCES = 10
SAFETY_K      = 1.5
SEED          = 42
N_CV_FOLDS    = 5


# -------------------- model factories (paper §V.D zoo) --------------------
def mk_lr():
    return LinearRegression()

def mk_knn(k=5):
    return Pipeline([("sc", StandardScaler()),
                     ("est", KNeighborsRegressor(n_neighbors=k))])

def mk_mlp():
    return Pipeline([("sc", StandardScaler()),
                     ("est", MLPRegressor(hidden_layer_sizes=(64, 32),
                                          max_iter=2000, random_state=SEED))])

def mk_rf():
    return RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)

def mk_lgbm():
    return LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                         min_data_in_leaf=2, verbose=-1, random_state=SEED)

def mk_ngb():
    return NGBRegressor(Dist=LogNormal, n_estimators=200, learning_rate=0.05,
                        verbose=False, random_state=SEED)


def make_zoo():
    """Return a list of (name, factory, train_log_target) tuples."""
    return [
        ("LR",      mk_lr,   True),
        ("kNN",     mk_knn,  True),
        ("MLP",     mk_mlp,  True),
        ("RF",      mk_rf,   True),
        ("LightGBM", mk_lgbm, True),
        ("NGBoost", mk_ngb,  False),  # raw target; LogNormal head; wrapped after fit
    ]


# -------------------- helpers --------------------
def cv_folds(n, k=N_CV_FOLDS, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    chunks = np.array_split(idx, min(k, n))
    for i, te in enumerate(chunks):
        tr = np.concatenate([c for j, c in enumerate(chunks) if j != i])
        if len(tr) and len(te):
            yield tr, te


def fit_one(name, factory, train_log_target, X_tr, y_log_tr):
    """Fit a single model class. Returns the (possibly wrapped) estimator."""
    if train_log_target:
        m = factory()
        m.fit(X_tr, y_log_tr)
        return m
    # NGBoost path: train on raw M, then wrap
    ngb = factory()
    ngb.fit(X_tr, np.exp(y_log_tr))
    return NGBLogMu(ngb)


def cv_residual_sigma(name, factory, train_log_target, X, y_log):
    """5-fold CV residual std in log space for one (model, view, bucket)."""
    n = len(X)
    if n < 2:
        return float("nan"), np.full(n, np.nan)
    pred = np.full(n, np.nan)
    for tr, te in cv_folds(n):
        try:
            m = fit_one(name, factory, train_log_target, X[tr], y_log[tr])
            pred[te] = m.predict(X[te])
        except Exception:
            # leave NaN; will be excluded from sigma
            continue
    mask = np.isfinite(pred)
    if mask.sum() < 2:
        return 0.5, pred  # conservative fallback
    sigma = float(np.std(y_log[mask] - pred[mask]))
    return sigma, pred


def train_view(view, X, y_log):
    """Train every model class in the zoo on (X, y_log) for one feature view.
    Returns list of (name, model, sigma) entries."""
    out = []
    for name, factory, train_log_target in make_zoo():
        sigma, _cv_pred = cv_residual_sigma(name, factory, train_log_target, X, y_log)
        try:
            full = fit_one(name, factory, train_log_target, X, y_log)
        except Exception as e:
            print(f"      [{view}] {name} FULL refit failed: {e}; skipping")
            continue
        out.append((name, full, sigma))
    return out


# -------------------- main --------------------
def main():
    print(f"loading {OURS_PATH.name} + {TRACE_PATH.name}")
    ours = pd.read_csv(OURS_PATH)
    trace = pd.read_csv(TRACE_PATH)
    on = pd.DataFrame({
        "workflow": ours["workflow"], "process": ours["process"],
        "a": pd.to_numeric(ours["a_bytes"], errors="coerce"),
        "c": pd.to_numeric(ours["c_bytes"], errors="coerce"),
        "M": pd.to_numeric(ours["M_peak_rss_bytes"], errors="coerce"),
        "runtime_sec": pd.to_numeric(ours["runtime_seconds"], errors="coerce"),
    }).dropna(subset=["a", "M"]).query("M > 0 and a >= 0").reset_index(drop=True)

    tn = pd.DataFrame({
        "workflow": "methylseq_naga",
        "process": trace["process"],
        "a": pd.to_numeric(trace["read_bytes"], errors="coerce"),
        "c": pd.to_numeric(trace["rchar"], errors="coerce"),
        "M": pd.to_numeric(trace["peak_rss"], errors="coerce"),
        "runtime_sec": pd.to_numeric(trace["realtime"], errors="coerce") / 1000.0,
    }).dropna(subset=["a", "c", "M"]).query("M > 0 and c > 0 and a >= 0").reset_index(drop=True)

    df = pd.concat([on, tn], ignore_index=True)
    df["runtime_sec"] = df["runtime_sec"].fillna(60.0)
    print(f"  rows: ours={len(on)}  methylseq_naga={len(tn)}  combined={len(df)}")
    print(f"  with c: {df['c'].notna().sum()} / {len(df)}")

    bucket_keys = (df.groupby(["workflow", "process"]).size()
                   .reset_index(name="n").query(f"n >= {MIN_INSTANCES}"))
    print(f"  buckets with n >= {MIN_INSTANCES}: {len(bucket_keys)}")

    to_pickle = {}
    summary_rows = []
    t0 = time.time()
    for ix, br in bucket_keys.iterrows():
        wf, proc, n = br.workflow, br.process, int(br.n)
        grp = df[(df.workflow == wf) & (df.process == proc)].reset_index(drop=True)
        a_log = np.log(grp["a"].values + EPS)
        M_log = np.log(grp["M"].values + EPS)
        c_mask = grp["c"].notna().values
        n_c = int(c_mask.sum())

        # Sizey view: all rows in bucket
        X_s = a_log.reshape(-1, 1)
        print(f"[{ix+1}/{len(bucket_keys)}] {wf} :: {proc[:50]:<50} n={n} n_c={n_c}")
        sizey_models = train_view("sizey", X_s, M_log)

        # Joint view: only c-present rows; only if c-present count >= MIN_INSTANCES
        joint_models = None
        if n_c >= MIN_INSTANCES:
            idx_c = np.flatnonzero(c_mask)
            X_j = np.column_stack([a_log[idx_c],
                                   np.log(grp["c"].values[idx_c] + EPS)])
            y_j = M_log[idx_c]
            joint_models = train_view("joint", X_j, y_j)

        bucket_key = f"{wf}::{proc}"
        to_pickle[bucket_key] = {
            "sizey_models": sizey_models,
            "joint_models": joint_models,
            "n":  n,
            "n_c": n_c,
            "safety_k": SAFETY_K,
        }

        summary_rows.append({
            "workflow": wf, "process": proc, "n": n, "n_c": n_c,
            "sizey_count": len(sizey_models),
            "joint_count": (0 if joint_models is None else len(joint_models)),
            "sizey_sigmas": ",".join(f"{s:.3f}" for _, _, s in sizey_models),
            "joint_sigmas": ("" if joint_models is None
                             else ",".join(f"{s:.3f}" for _, _, s in joint_models)),
        })

    elapsed = time.time() - t0
    print(f"\ntrained {len(to_pickle)} buckets in {elapsed:.1f}s")

    # ---- write artifacts
    PKL  = MODEL_DIR / "per_task_full.pkl"
    META = MODEL_DIR / "per_task_full.json"
    SUMM = MODEL_DIR / "per_task_full_summary.csv"

    with open(PKL, "wb") as f:
        pickle.dump(to_pickle, f)

    json_meta = {
        k: {
            "n": v["n"],
            "n_c": v["n_c"],
            "safety_k": v["safety_k"],
            "sizey_models": [{"name": n, "sigma": s} for n, _, s in v["sizey_models"]],
            "joint_models": (None if v["joint_models"] is None
                             else [{"name": n, "sigma": s} for n, _, s in v["joint_models"]]),
        }
        for k, v in to_pickle.items()
    }
    with open(META, "w") as f:
        json.dump(json_meta, f, indent=2)
    pd.DataFrame(summary_rows).to_csv(SUMM, index=False)

    print(f"wrote {PKL}  ({PKL.stat().st_size:,} bytes)")
    print(f"wrote {META}")
    print(f"wrote {SUMM}")

    # ---- sanity counts
    n_buckets        = len(to_pickle)
    n_sizey_complete = sum(1 for v in to_pickle.values() if len(v["sizey_models"]) == 6)
    n_joint_complete = sum(1 for v in to_pickle.values()
                           if v["joint_models"] is not None and len(v["joint_models"]) == 6)
    n_iwd_joint_none = sum(1 for k, v in to_pickle.items()
                           if k.startswith("iwd::") and v["joint_models"] is None)
    print("\n=== sanity ===")
    print(f"  buckets total:                    {n_buckets}")
    print(f"  buckets with 6 sizey_models:      {n_sizey_complete}")
    print(f"  buckets with 6 joint_models:      {n_joint_complete}")
    print(f"  iwd buckets with joint_models=None: {n_iwd_joint_none}")


if __name__ == "__main__":
    main()
