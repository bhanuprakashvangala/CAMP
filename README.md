# CAMP - Consumption-Aware Memory Prediction

Artifact for the IEEE Cluster 2026 paper *Consumption-Aware Memory
Prediction for Scientific Workflow Tasks*. CAMP predicts per-task peak
memory from the requested input size `a(t)` and the byte-granular
consumed size `c(t)`, fits a per-bucket model zoo (bucket = `(workflow,
task-type)`), and selects which tasks receive the `c`-audit via an
active-learning gate.

## Layout

```text
workflows/   Per-workflow rerun inputs, configs, datasets, and validation CSVs
audit/       Selective auditing, both lanes
  ebpf/        online lane: selective wrapper, eBPF tracer, merge helpers
  strace/      offline lane: replay scripts, per-task table builders, task-plan filter
code/        Scripts that produce the models, results, and figures
  experiment_1/   6-class zoo, Sizey vs Joint feature views (deterministic)
  experiment_2/   probabilistic zoo (NGBoost, Q-LGBM, Bayesian Ridge) on eBPF lane
  experiment_3/   round-based active learning + IPW retraining
  reruns/         paper-faithful reimplementation (max-aggregation deploy, NGBoost gate)
  gating/         paper-faithful gating entry points and task-plan generation
data/        Input traces: joined per-task table, methylseq trace, pyradiomics
             task metrics, and the eBPF audit traces (data/ebpf/)
models/      Trained per-bucket zoo (per_task_full.pkl) and Exp-2 models (models_exp2.pkl)
results/     Derived tables the figures read (predictions, AL curves, comparison, sanity)
figures/     Every paper figure, PNG + PDF (figures/README.md maps each to its script)
```

## Navigation

- [workflows/README.md](workflows/README.md) - workflow matrix, per-workflow run commands, dataset provenance. Experiment 1: `eager`, `rnaseq`, `mag`, `mag_karlsson`, `chipseq`, `methylseq`, `pyradiomics`; Experiment 2: `bowtie`, `minimap`, `mcmicro`.
- [code/gating/README.md](code/gating/README.md) - the gating commands that turn predictions into a `task_plan.csv`.
- [audit/ebpf/README.md](audit/ebpf/README.md) - the selective eBPF lane (online) used by Experiment 2.
- [code/README.md](code/README.md) - ML-side scripts, preserved prediction CSVs, and reproduction scope.

## Reviewer sequence

End-to-end validation path:

1. Reproduce one preserved workflow run from [workflows/README.md](workflows/README.md).
2. Build the workflow-level per-task table from that rerun.
3. Predict memory from static task information (`workflow`, `process`, `a_bytes`). The reviewer-facing handoff uses the preserved prediction CSVs documented in [code/README.md](code/README.md).
4. Generate the gating plan from the predictions with [code/gating/experiment1_gating_logic_paper.py](code/gating/experiment1_gating_logic_paper.py) (point) or [code/gating/experiment2_gating_logic_paper.py](code/gating/experiment2_gating_logic_paper.py) (distribution).
5. Apply the gate only to tasks marked `Audit`: Experiment 2 tasks receive `eBPF` collection through the selective wrapper in [audit/ebpf](audit/ebpf); Experiment 1 tasks receive `strace` collection after filtering the rerun table with [audit/strace/filter_after_by_task_plan.py](audit/strace/filter_after_by_task_plan.py).
6. Merge the audited dynamic features back into the workflow task table - this adds the `c_bytes` used by the enhanced model view.
7. Evaluate CAMP against the preserved workflow and audit outputs: predict first, gate, recover `c`, and use the enhanced features to improve the next prediction.

This validates both the problem (static-only descriptors miss consumption behavior) and the solution (predict first, audit selectively, recover `c`, improve prediction while avoiding full auditing).

## Key numbers

- Per-model calibration (held-out): RF Joint 1.43% MAPE, LightGBM Joint 2.93%, NGBoost Joint 3.23% (`figures/fig9_per_model_calibration`).
- Deployment max-aggregation vs LGBM-only routing: OOMs 28 -> 1 at a modest wastage increase (`results/compare_old_vs_new.csv`).
- Active learning: gate vs random vs full-audit oracle across budgets 0.05-0.30 (`results/results_active_learning.csv`, `figures/fig_e3_al_curves`).
