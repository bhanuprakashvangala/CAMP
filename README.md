# CAMP - Consumption-Aware Memory Prediction

Artifact for the IEEE Cluster 2026 paper *Consumption-Aware Memory
Prediction for Scientific Workflow Tasks*. CAMP predicts per-task peak
memory from the requested input size `a(t)` and the byte-granular
consumed size `c(t)`, fits a per-bucket model zoo (bucket = `(workflow,
task-type)`), and selects which tasks receive the `c`-audit via an
active-learning gate.

## Layout

```text
figures/    Every bundled figure used in the paper, PNG + PDF
code/       Scripts that produce the models, results, and figures
  experiment_1/   6-class zoo, Sizey vs Joint feature views (deterministic)
  experiment_2/   probabilistic zoo (NGBoost, Q-LGBM, Bayesian Ridge) on eBPF lane
  experiment_3/   round-based active learning + IPW retraining
  reruns/         paper-faithful reimplementation (max-aggregation deploy,
                  NGBoost gate); see code/reruns/REPORT.md
data/       Input traces - joined per-task table, methylseq trace,
            pyradiomics task metrics, and the eBPF audit traces (data/ebpf/)
models/     Trained per-bucket zoo (per_task_full.pkl) and the Exp-2
            probabilistic models (models_exp2.pkl)
results/    Derived tables the figures read (predictions, AL curves,
            old-vs-new deployment comparison, sanity report)
figures/README.md  Maps each bundled figure to the script that generates it
```

## Reviewer Sequence

For artifact review, the intended end-to-end validation path is:

1. Reproduce one of the preserved workflow runs from [workflows/README.md](workflows/README.md).
2. Build the workflow-level per-task table from that rerun.
3. Use the workflow output as input to the memory-prediction stage.
   The first prediction stage uses static task information such as `workflow`, `process`, and `a_bytes`.
4. Generate the gating plan from the prediction outputs.
   Paper-faithful gating entry points are in [gating_logic/experiment1_gating_logic_paper.py](gating_logic/experiment1_gating_logic_paper.py) and [gating_logic/experiment2_gating_logic_paper.py](gating_logic/experiment2_gating_logic_paper.py).
5. Apply the gating decision only to the tasks marked for audit.
   For Experiment 2, those tasks are the ones that should receive `eBPF`-based dynamic feature collection through the selective wrapper.
   For the replay-based Experiment 1 workflows, those tasks are the ones that should receive `strace`-based dynamic feature collection after the rerun task table is filtered by the generated `task_plan.csv`.
6. Merge the audited dynamic features back into the workflow task table.
   This is the step that adds the `c_bytes` information used by the enhanced model view.
7. Re-run the prediction stage with the enriched task table and evaluate the CAMP solution.
   In the paper workflow, the static-input prediction is used first, the gate selects which tasks to audit, and the audited `c` information is then used to improve the next prediction/training stage.

At a high level, this sequence validates both the problem and the solution discussed in the paper:

- the memory-prediction problem: static-only task descriptors can miss important consumption behavior
- the CAMP solution: predict first, audit selectively, recover `c`, and use the enhanced features to improve memory prediction while avoiding full auditing

## Workflow Guide

Please use the workflow folders as the reviewer handoff for detailed run instructions, dataset provenance, compact output CSVs, and workflow-specific prerequisites.

- Start with [workflows/README.md](workflows/README.md) for the high-level workflow matrix and command sequence.
- For Experiment 1 workflow details, use the individual folders under [workflows](workflows), especially `eager`, `rnaseq`, `mag`, `mag_karlsson`, `chipseq`, `methyl_seq`, and `pyradiomics`.
- For Experiment 2 workflow details, use [workflows/bowtie](workflows/bowtie), [workflows/minimap](workflows/minimap), and [workflows/mcmicro](workflows/mcmicro).
- For selective audit implementation details, use [EBPF/README.md](EBPF/README.md) together with [gating_logic/README.md](gating_logic/README.md).

## Reproduce

Requires Python 3.12 with `numpy pandas scikit-learn lightgbm ngboost matplotlib scipy`.

```bash
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

Paths in the scripts assume the original `IEEE_CLUSTER_MAIN/` working tree; adjust the
`REPO` and data constants at the top of each script to point at this artifact's `data/`
and `models/` directories.

## Workflow Outputs

If a reviewer starts from an individual workflow rerun, please use
[workflows/README.md](workflows/README.md) as the handoff guide into the
memory-prediction artifact. For `eager`, `rnaseq`, `mag`, `mag_karlsson`,
and `chipseq`, the workflow-side file that connects into the point-based
ML code is the reconstructed `task_metrics_rerun.csv`. In the preserved
snapshot, each workflow's `task_metrics_recorded.csv` matches that
workflow's exact row subset in [data/all_workflows.csv](data/all_workflows.csv),
so a reviewer can substitute a newly generated `task_metrics_rerun.csv`
for the corresponding workflow slice before running the ML scripts here.

`pyradiomics` and `methylseq` are ingested separately in the current
artifact: the ML code uses
[data/task_metrics_pyradiomics_32k_detailed.csv](data/task_metrics_pyradiomics_32k_detailed.csv)
for PyRadiomics and [data/trace_methylseq.csv](data/trace_methylseq.csv)
for methylseq.

## Key Numbers

- Per-model calibration (held-out): RF Joint 1.43% MAPE, LightGBM Joint 2.93%, NGBoost Joint 3.23% (`figures/fig9_per_model_calibration`).
- Deployment max-aggregation vs LGBM-only routing: OOMs 28 -> 1 at a modest wastage increase (`results/compare_old_vs_new.csv`).
- Active learning: gate vs random vs full-audit oracle across budgets 0.05-0.30 (`results/results_active_learning.csv`, `figures/fig_e3_al_curves`).
