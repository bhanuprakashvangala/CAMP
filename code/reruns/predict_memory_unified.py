#!/usr/bin/env python3
"""
reruns/predict_memory_unified.py <workflow> <process> <a_bytes> [<c_bytes>]

Max-aggregation over all retained models in the chosen feature view:

    m_safe(t) = max_i  exp( mu_hat_i(x(t)) + safety_k * sigma_{b,i,v} )

view v is selected by audit status:
    - c provided           -> Joint  view  X = [log a, log c]
    - c omitted (unaudited) -> Sizey view  X = [log a]

For iwd buckets, joint_models is None; we always fall back to sizey.
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
import numpy as np


def predict_safe(info: dict, a_bytes: float, c_bytes: float | None,
                 eps: float = 1.0):
    """Return (M_safe_bytes, contributions[name -> safe_bytes], view_used)."""
    if c_bytes is None or info.get("joint_models") is None:
        models_view = info["sizey_models"]
        x = np.array([[np.log(a_bytes + eps)]])
        view_used = "sizey"
    else:
        models_view = info["joint_models"]
        x = np.array([[np.log(a_bytes + eps), np.log(c_bytes + eps)]])
        view_used = "joint"
    safety_k = float(info["safety_k"])

    contributions = {}
    safe_candidates = []
    for name_i, model_i, sigma_i in models_view:
        try:
            mu_i = float(np.asarray(model_i.predict(x)).reshape(-1)[0])
        except Exception as e:
            sys.stderr.write(f"warn: predict failed for {name_i}: {e}\n")
            continue
        safe_i = float(np.exp(mu_i + safety_k * float(sigma_i)))
        contributions[name_i] = safe_i
        safe_candidates.append(safe_i)

    if not safe_candidates:
        raise RuntimeError("no candidate models produced a prediction")
    M_safe = float(max(safe_candidates))
    return M_safe, contributions, view_used


def main():
    if len(sys.argv) not in (4, 5):
        sys.exit("usage: predict_memory_unified.py <workflow> <process> <a_bytes> [<c_bytes>]")
    wf, proc, a = sys.argv[1], sys.argv[2], float(sys.argv[3])
    c = float(sys.argv[4]) if len(sys.argv) == 5 else None

    pkl = Path(__file__).resolve().parent / "models_unified" / "per_task_full.pkl"
    with open(pkl, "rb") as f:
        M = pickle.load(f)

    key = f"{wf}::{proc}"
    if key not in M:
        sys.stderr.write(f"no model for {key}; defaulting to 4 GB\n")
        print("--mem=4096M")
        sys.exit(0)

    info = M[key]
    M_safe, contributions, view = predict_safe(info, a, c)

    sys.stderr.write(f"view={view}  candidates:\n")
    for name, val in sorted(contributions.items(), key=lambda kv: -kv[1]):
        sys.stderr.write(f"  {name:<10} safe={val/(1024**2):.1f} MB\n")
    sys.stderr.write(f"max-aggregated safe={M_safe/(1024**2):.1f} MB\n")

    print(f"--mem={int(np.ceil(M_safe / (1024**2)))}M")


if __name__ == "__main__":
    main()
