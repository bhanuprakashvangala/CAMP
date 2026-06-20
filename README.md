# CAMP — Consumption-Aware Memory Prediction

Artifact for the IEEE Cluster 2026 paper *Consumption-Aware Memory
Prediction for Scientific Workflow Tasks*. CAMP predicts per-task peak
memory from the requested input size `a(t)` and the byte-granular
consumed size `c(t)`, fits a per-bucket model zoo (bucket = `(workflow,
task-type)`), and selects which tasks receive the `c`-audit via an
active-learning gate.

## Layout

```
figures/    Every figure used in the paper, PNG + PDF
code/       Scripts that produce the models, results, and figures
  experiment_1/   6-class zoo, Sizey vs Joint feature views (deterministic)
  experiment_2/   probabilistic zoo (NGBoost, Q-LGBM, Bayesian Ridge) on eBPF lane
  experiment_3/   round-based active learning + IPW retraining
  reruns/         paper-faithful reimplementation (max-aggregation deploy,
                  NGBoost gate); see code/reruns/REPORT.md
data/       Input traces — joined per-task table, methylseq trace,
            pyradiomics task metrics, and the eBPF audit traces (data/ebpf/)
models/     Trained per-bucket zoo (per_task_full.pkl) and the Exp-2
            probabilistic models (models_exp2.pkl)
results/    Derived tables the figures read (predictions, AL curves,
            old-vs-new deployment comparison, sanity report)
FIGURES.md  Maps each paper figure to the script that generates it
```

## Reproduce

Requires Python 3.12 with `numpy pandas scikit-learn lightgbm ngboost
matplotlib scipy`.

```
# train the per-bucket zoo  -> models/per_task_full.pkl
python code/reruns/train_full.py

# deployment-time safe allocation for one task
python code/reruns/predict_memory_unified.py <workflow> <process> <a_bytes> [<c_bytes>]

# active learning curves     -> results/results_active_learning.csv + figures
python code/reruns/run_active_learning.py

# headline calibration / wastage / OOM figures
python code/experiment_1/fig9_regen.py
python code/experiment_1/fig_method_summary.py
```

Paths in the scripts assume the original `IEEE_CLUSTER_MAIN/` working
tree; adjust the `REPO`/data constants at the top of each script to point
at this artifact's `data/` and `models/` directories.

## Key numbers

- Per-model calibration (held-out): RF Joint 1.43% MAPE, LightGBM Joint
  2.93%, NGBoost Joint 3.23% (`figures/fig9_per_model_calibration`).
- Deployment max-aggregation vs LGBM-only routing: OOMs 28 -> 1 at a
  modest wastage increase (`results/compare_old_vs_new.csv`).
- Active learning: gate vs random vs full-audit oracle across budgets
  0.05–0.30 (`results/results_active_learning.csv`,
  `figures/fig_e3_al_curves`).
