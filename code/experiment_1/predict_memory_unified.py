#!/usr/bin/env python3
"""
predict_memory_unified.py <workflow> <process> <a_bytes> [<c_bytes>]

If <c_bytes> is given (audited path) -> uses joint LightGBM(a, c).
If omitted (unaudited path)         -> falls back to Sizey LightGBM(a).

Safety = predicted log M + safety_k * per-method residual std (in log space).
Prints: --mem=NNNM
"""
import sys, pickle
from pathlib import Path
import numpy as np
if len(sys.argv) not in (4, 5):
    sys.exit("usage: predict_memory_unified.py <workflow> <process> <a_bytes> [<c_bytes>]")
wf, proc, a = sys.argv[1], sys.argv[2], float(sys.argv[3])
c = float(sys.argv[4]) if len(sys.argv) == 5 else None
with open(Path(__file__).parent / "models_unified" / "per_task.pkl", "rb") as f:
    M = pickle.load(f)
key = f"{wf}::{proc}"
if key not in M:
    sys.stderr.write(f"no model for {key}; defaulting to 4 GB\n")
    print("--mem=4096M"); sys.exit(0)
info = M[key]; EPS = 1.0
if c is None:
    M_log = float(info["sizey"].predict(np.array([[np.log(a + EPS)]]))[0])
    cushion = info["resid_sizey"]
else:
    M_log = float(info["joint"].predict(np.array([[np.log(a + EPS), np.log(c + EPS)]]))[0])
    cushion = info["resid_joint"]
M_safe = float(np.exp(M_log + info["safety_k"] * cushion))
print(f"--mem={int(np.ceil(M_safe / (1024**2)))}M")
