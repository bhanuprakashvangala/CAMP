"""For each held-out bucket, find which model wins (lowest MAPE) — both variants."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd, sys
from pathlib import Path

REPO = Path("/shared/training")
EPS, MIN, SEED = 1.0, 10, 42

with open(REPO/"models_unified/per_task_paper.pkl", "rb") as f:
    models = pickle.load(f)

ours = pd.read_csv(REPO/"all_workflows.csv")
on = pd.DataFrame({"workflow":ours["workflow"], "process":ours["process"],
    "a":pd.to_numeric(ours["a_bytes"],errors="coerce"),
    "c":pd.to_numeric(ours["c_bytes"],errors="coerce"),
    "M":pd.to_numeric(ours["M_peak_rss_bytes"],errors="coerce"),
}).dropna(subset=["a","M"]).query("M>0 and a>=0").reset_index(drop=True)
on = on[on["workflow"] != "methylseq"]
sys.path.insert(0, str(REPO))
from load_pyradiomics import load_pyradiomics
pn = load_pyradiomics(str(REPO/"pyradiomics_32k.csv"))[["workflow","process","a","c","M"]]
df = pd.concat([on, pn], ignore_index=True)

test_rows = []
for (wf, proc), grp in df.groupby(["workflow","process"]):
    if len(grp) < MIN: continue
    g = grp.sample(frac=1, random_state=SEED).reset_index(drop=True)
    cut = int(len(g) * 0.8)
    te = g.iloc[cut:].copy(); te["bucket"] = f"{wf}::{proc}"
    test_rows.append(te)
test_df = pd.concat(test_rows, ignore_index=True)

MODELS = ["lr","knn","mlp","rf","lgbm","ngb"]

def predict_for_bucket(bucket, model_name, variant):
    g = test_df[test_df["bucket"] == bucket].reset_index(drop=True)
    info = models.get(bucket)
    if info is None: return None, None
    key = f"{model_name}_{variant}"
    m_obj = info["fits"].get(key)
    if m_obj is None: return None, None
    a_log = np.log(g["a"].values + EPS)
    c_present = g["c"].notna().values
    if variant == "joint":
        if not c_present.any(): return None, None
        X = np.column_stack([a_log[c_present],
                              np.log(g["c"].values[c_present]+EPS)])
        actual = g["M"].values[c_present]
    else:
        X = a_log.reshape(-1,1)
        actual = g["M"].values
    try:
        from ngboost import NGBRegressor
        if isinstance(m_obj, NGBRegressor):
            p = np.exp(np.log(np.maximum(1.0, m_obj.pred_dist(X).mean())))
        else:
            p = np.exp(m_obj.predict(X))
    except Exception:
        return None, None
    mape = float(np.mean(np.abs(p - actual) / actual) * 100)
    return mape, len(actual)

# === Per-bucket winners ===
print(f"{'bucket':<55}{'n':>6}{'n_c':>6}", end="")
for v in ["sizey","joint"]:
    for m in MODELS:
        print(f"  {m+('_S' if v=='sizey' else '_J'):>8}", end="")
print(f"  {'best Sizey':>14}  {'best Joint':>14}")
print("-"*200)

bucket_winners = {"sizey":{}, "joint":{}}
for bucket in sorted(test_df["bucket"].unique()):
    g = test_df[test_df["bucket"]==bucket]
    n = len(g); n_c = int(g["c"].notna().sum())
    line = f"{bucket[:53]:<55}{n:>6}{n_c:>6}"

    sizey_results, joint_results = {}, {}
    for v in ["sizey","joint"]:
        for m in MODELS:
            mape, _ = predict_for_bucket(bucket, m, v)
            if v=="sizey": sizey_results[m] = mape
            else:           joint_results[m] = mape
            cell = f"{mape:6.2f}%" if mape is not None else "   --  "
            line += f"  {cell:>8}"

    valid_s = {k:v for k,v in sizey_results.items() if v is not None}
    valid_j = {k:v for k,v in joint_results.items() if v is not None}
    best_s = min(valid_s, key=valid_s.get) if valid_s else "—"
    best_j = min(valid_j, key=valid_j.get) if valid_j else "—"
    line += f"  {best_s.upper()+(f' ({valid_s[best_s]:.2f}%)' if best_s != '—' else ''):>14}"
    line += f"  {best_j.upper()+(f' ({valid_j[best_j]:.2f}%)' if best_j != '—' else ''):>14}"
    print(line)
    bucket_winners["sizey"][bucket] = best_s
    bucket_winners["joint"][bucket] = best_j

# === Tally ===
print()
print("="*100)
print("WINNER TALLY")
print("="*100)
for v in ["sizey","joint"]:
    counts = pd.Series(list(bucket_winners[v].values())).value_counts()
    print(f"\n{v.upper()} variant — bucket wins per model:")
    for m, c in counts.items():
        print(f"  {m.upper():<6}  {c:>3} buckets")
