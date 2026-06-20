#!/usr/bin/env python3
"""
reruns/compare_old_vs_new.py — quantify the change in §VI numbers from
switching to max-aggregated 6-class zoo.

Loads both pickles:
  - old: experiment_1/models_unified/per_task.pkl  (LGBM-only routing)
  - new: reruns/models_unified/per_task_full.pkl   (max over 6-class zoo)

Re-walks every row of the joined dataset, computes the safe allocation
that each pickle would have produced (Sizey if no c, Joint if c present),
and aggregates wastage GB-h and OOMs by workflow and overall.
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
NEW_PKL  = REPO / "models_unified" / "per_task_full.pkl"
# Old deployment-time pickle from experiment_1.  Two are available:
#   per_task_paper.pkl     — has 6-class fits+sigmas, but Sizey-only and
#                            limited to iwd+pyradiomics.
#   per_task_no_methylseq.pkl — has the schema that the original
#                            predict_memory_unified.py expects (sizey/joint
#                            LGBM + resid_sizey/joint).
OLD_PKL  = REPO.parent / "experiment_1" / "models_unified" / "per_task_no_methylseq.pkl"
OURS_CSV = REPO.parent / "experiment_1" / "joined" / "all_workflows.csv"
TRACE_CSV = REPO.parent / "experiment_1" / "trace_methylseq.csv"
OUT_CSV  = REPO / "models_unified" / "compare_old_vs_new.csv"

EPS = 1.0


def load_data():
    ours = pd.read_csv(OURS_CSV)
    trace = pd.read_csv(TRACE_CSV)
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
    return df


def old_safe(info, a, c):
    if c is None or info.get("joint") is None:
        x = np.array([[np.log(a + EPS)]])
        mu = float(info["sizey"].predict(x)[0])
        return float(np.exp(mu + info["safety_k"] * info["resid_sizey"]))
    x = np.array([[np.log(a + EPS), np.log(c + EPS)]])
    mu = float(info["joint"].predict(x)[0])
    return float(np.exp(mu + info["safety_k"] * info["resid_joint"]))


def new_safe(info, a, c):
    if c is None or info.get("joint_models") is None:
        x = np.array([[np.log(a + EPS)]])
        models = info["sizey_models"]
    else:
        x = np.array([[np.log(a + EPS), np.log(c + EPS)]])
        models = info["joint_models"]
    safety_k = float(info["safety_k"])
    cands = []
    for name, m, sigma in models:
        try:
            mu = float(np.asarray(m.predict(x)).reshape(-1)[0])
        except Exception:
            continue
        cands.append(np.exp(mu + safety_k * float(sigma)))
    return float(max(cands)) if cands else float("nan")


def main():
    with open(NEW_PKL, "rb") as f:
        M_new = pickle.load(f)
    with open(OLD_PKL, "rb") as f:
        M_old = pickle.load(f)

    df = load_data()
    rows = []
    for _, r in df.iterrows():
        key = f"{r.workflow}::{r.process}"
        if key not in M_new or key not in M_old:
            continue
        info_new = M_new[key]
        info_old = M_old[key]
        c = float(r.c) if pd.notna(r.c) else None
        try:
            s_old = old_safe(info_old, float(r.a), c)
            s_new = new_safe(info_new, float(r.a), c)
        except Exception:
            continue
        rows.append({
            "workflow": r.workflow, "process": r.process,
            "a": r.a, "c": r.c, "M": r.M,
            "runtime_sec": r.runtime_sec,
            "safe_old_bytes": s_old, "safe_new_bytes": s_new,
            "OOM_old": int(s_old < r.M), "OOM_new": int(s_new < r.M),
            "waste_old_GBh": max(s_old - r.M, 0) / (1024**3) * r.runtime_sec / 3600,
            "waste_new_GBh": max(s_new - r.M, 0) / (1024**3) * r.runtime_sec / 3600,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(out):,} rows)")

    print("\n=== overall ===")
    print(f"{'metric':<22} {'old':>16} {'new':>16}")
    for col in ["waste_old_GBh", "waste_new_GBh"]:
        pass
    print(f"{'wastage_GBh_total':<22} {out.waste_old_GBh.sum():>16,.2f} {out.waste_new_GBh.sum():>16,.2f}")
    print(f"{'OOM_count_total':<22} {int(out.OOM_old.sum()):>16d} {int(out.OOM_new.sum()):>16d}")

    print("\n=== per workflow ===")
    g = out.groupby("workflow").agg(
        n=("M", "size"),
        waste_old=("waste_old_GBh", "sum"),
        waste_new=("waste_new_GBh", "sum"),
        oom_old=("OOM_old", "sum"),
        oom_new=("OOM_new", "sum"),
    ).reset_index().sort_values("waste_new", ascending=False)
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
