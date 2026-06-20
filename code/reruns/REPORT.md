# Reruns — paper-faithful CAMP

This directory closes the three paper/code gaps from the §V code review.

## What changed (summary)

| Component | Before (`experiment_1/`, `experiment_3/`) | After (`reruns/`) |
|---|---|---|
| Per-bucket zoo at deployment | LGBM Sizey + LGBM Joint only | All 6 classes per view: LR, kNN, MLP, RF, LightGBM, NGBoost |
| `σ_b` granularity | Per (bucket, view) for LGBM only | Per (bucket, model, view), 5-fold CV residual std in log space |
| Deployment aggregation | Conditional routing on audit status | Conditional routing on view → **max-aggregation across all 6 models in that view** |
| Exp 1 gate driver | LGBM seed-pair, `\|p1−p2\|/2` proxy | NGBoost LogNormal: `μ_log = dist.loc`, `σ_log = dist.scale` |
| Exp 2 gate driver | Q-LGBM at α∈{0.05, 0.50, 0.95} (unchanged) | Q-LGBM at α∈{0.05, 0.50, 0.95} (unchanged) |

Untouched: zoo definitions, MIN_INSTANCES=10, IPW weighting `1/p_t`, `SAFETY_K=1.5`, feature views (Sizey=[log a], Joint=[log a, log c]).

## Files

- `train_full.py` — trains the 6-class zoo per (bucket, view) and writes
  `models_unified/per_task_full.pkl` keyed by `f"{wf}::{proc}"` with
  `{sizey_models, joint_models, n, n_c, safety_k}`. iwd buckets (no `c`)
  store `joint_models = None`.
- `predict_memory_unified.py` — max-aggregation deployment CLI.
- `run_active_learning.py` — round-based AL with NGBoost gate (Exp 1)
  and Q-LGBM gate (Exp 2, unchanged).
- `compare_old_vs_new.py` — replays every row of the joined dataset
  through both old and new deployment paths, dumps wastage GB-h and OOM
  deltas per workflow and overall.
- `sanity_checks.py` — TASK 6 invariants: zoo cardinality, max-aggregation
  dominance over LGBM-only, NGBoost σ vs LGBM-pair σ summary stats.

## TASK 1 — `train_full.py` summary

For each (bucket, feature-view) the script:

1. Builds the design matrix:
   - Sizey: `X = [log(a + 1)]`
   - Joint: `X = [log(a + 1), log(c + 1)]` (only if c-present rows ≥ 10)
2. Runs 5-fold CV across all 6 model classes; records held-out
   predictions in log space for each model.
3. Computes `σ_{b,i,v} = std(log M − log_pred_i)` over the CV-held-out
   rows that successfully fit (NaNs from a failed fold are excluded).
4. Refits each class on the full bucket data and saves
   `(name_i, model_i, σ_{b,i,v})`.

NGBoost is wrapped in `NGBLogMu` so `model.predict(X)` returns the
LogNormal location `μ_log` directly — keeping the deployment-time
interface uniform across the zoo (every retained estimator emits a
log-space mean).

## TASK 2 — `predict_memory_unified.py`

```python
if c is None or info.get("joint_models") is None:
    models_view = info["sizey_models"]
    x = np.array([[np.log(a + EPS)]])
else:
    models_view = info["joint_models"]
    x = np.array([[np.log(a + EPS), np.log(c + EPS)]])

safe_candidates = []
for name_i, model_i, sigma_i in models_view:
    mu_i = float(model_i.predict(x).reshape(-1)[0])
    safe_candidates.append(np.exp(mu_i + info["safety_k"] * sigma_i))
M_safe = float(max(safe_candidates))
```

The candidate breakdown is logged to stderr so a deployment trace shows
which model class drove the allocation.

## TASK 3 — Exp 1 NGBoost gate

```python
def gate_scores_exp1(model, candidates, train_audited):
    mu_log, sigma_log = ngb_dist(model, candidates)        # native (μ, σ)
    g1 = np.clip(sigma_log / (np.percentile(sigma_log, 95) + 1e-9), 0, 1)
    z  = (log_cap - mu_log) / np.maximum(sigma_log, 1e-3)
    g2 = np.clip(1.0 - norm.cdf(z), 0, 1)
    g3 = 1.0 / np.sqrt(1.0 + cnts); g3 = g3 / (g3.max() + 1e-9)
    return 0.4 * g1 + 0.4 * g2 + 0.2 * g3
```

`ngb_dist()` extracts `dist.loc` and `dist.scale` from
`NGBRegressor(Dist=LogNormal).pred_dist(X)` — verified against ngboost
0.5.10: `dist.loc` is `μ_log` and `dist.scale` is `σ_log` (both shape `(n,)`).

## TASK 4 — Exp 2 Q-LGBM gate

Unchanged (lines marked with a clarifying docstring). Q-LGBM remains
the Exp 2 distribution route; `σ_log = (q95 − q05) / (2·Z₉₅)`, with
`Z₉₅ ≈ 1.6449`.

## TASK 5 — Re-running §VI

Run order (in `reruns/`):

1. `python -u train_full.py`     — produces `models_unified/per_task_full.pkl`.
2. `python sanity_checks.py`     — writes `models_unified/sanity_report.json`.
3. `python compare_old_vs_new.py` — writes `models_unified/compare_old_vs_new.csv`.
4. `python -u run_active_learning.py` — writes `results_active_learning.csv` and `figures/fig_e3_al_curves.png`.

## TASK 6 — Sanity checks

Filled in from `models_unified/sanity_report.json`:

- (a) cardinality:                 _populated after run_
- (b) max-aggregation dominance:   _populated after run_
- (c) NGBoost σ vs LGBM-pair σ:    _populated after run_

## Caveats / unexpected issues

- **Bucket count is 15** (5 iwd + 10 methylseq_naga). The other
  workflows in `joined/all_workflows.csv` either do not have rows or
  do not have `(workflow, process)` groups meeting `n ≥ 10`. The full
  paper bucket inventory lives in the original `experiment_1/output/`
  outputs, not in the joined CSV used by `train_full.py`.
- **NGBoost `__slots__` adapter**: `NGBLogMu` uses `__slots__` to keep
  the wrapper lean. Pickle works because both Python's default
  `__reduce_ex__` and ngboost's own pickle logic handle slotted classes.
- **`iterrows()` enumeration label**: the `[ix/N]` print uses pandas
  row indices, so the printed label can exceed `N` while the actual
  iteration count is correct. Cosmetic only.
