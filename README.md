# CAMP - Consumption-Aware Memory Prediction

Artifact for the IEEE Cluster 2026 paper *Consumption-Aware Memory
Prediction for Scientific Workflow Tasks*. CAMP predicts per-task peak
memory from the requested input size `a(t)` and the byte-granular
consumed size `c(t)`, fits a per-bucket model zoo (bucket = `(workflow,
task-type)`), and selects which tasks receive the `c`-audit via an
active-learning gate.

## Layout

```text
workflows/   Reviewer-facing workflow rerun inputs, commands, and validation CSVs
gating_logic/ Paper-faithful gating entry points and task-plan generation commands
EBPF/        Shared selective eBPF wrapper, tracer, and helper utilities
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
code/README.md     Reviewer-facing ML reproduction notes and current scope
```

## Start Here

For artifact review, the intended reading and execution order is:

1. Start with [workflows/README.md](workflows/README.md) and reproduce one workflow run.
2. Use [code/README.md](code/README.md) to understand the ML-side handoff and the current preserved-prediction scope.
3. Run the paper-faithful gating commands from [gating_logic/README.md](gating_logic/README.md).
4. For selective audit implementation details, use [EBPF/README.md](EBPF/README.md) together with the workflow-specific commands.

## Reviewer Sequence

For artifact review, the intended end-to-end validation path is:

1. Reproduce one of the preserved workflow runs from [workflows/README.md](workflows/README.md).
2. Build the workflow-level per-task table from that rerun.
3. Use the workflow output as input to the memory-prediction stage.
   The first prediction stage uses static task information such as `workflow`, `process`, and `a_bytes`.
   In the current artifact snapshot, the reviewer-facing gating handoff uses the preserved prediction CSVs documented in [code/README.md](code/README.md).
4. Generate the gating plan from the prediction outputs.
   Paper-faithful gating entry points are in [gating_logic/experiment1_gating_logic_paper.py](gating_logic/experiment1_gating_logic_paper.py) and [gating_logic/experiment2_gating_logic_paper.py](gating_logic/experiment2_gating_logic_paper.py).
5. Apply the gating decision only to the tasks marked for audit.
   For Experiment 2, those tasks are the ones that should receive `eBPF`-based dynamic feature collection through the selective wrapper.
   For the replay-based Experiment 1 workflows, those tasks are the ones that should receive `strace`-based dynamic feature collection after the rerun task table is filtered by the generated `task_plan.csv`.
6. Merge the audited dynamic features back into the workflow task table.
   This is the step that adds the `c_bytes` information used by the enhanced model view.
7. Evaluate the CAMP solution against the preserved workflow and audit outputs.
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
- For ML-side scripts, preserved prediction CSVs, and current reproduction scope, use [code/README.md](code/README.md).

## Code Guide

Use [code/README.md](code/README.md) for the ML-side artifact notes:

- which `code/` subdirectories correspond to Experiments 1, 2, and 3
- which prediction CSVs are preserved and used by the gating stage
- which scripts are directly runnable from this artifact snapshot
- where the current fresh prediction-regeneration limitation begins

## Key Numbers

- Per-model calibration (held-out): RF Joint 1.43% MAPE, LightGBM Joint 2.93%, NGBoost Joint 3.23% (`figures/fig9_per_model_calibration`).
- Deployment max-aggregation vs LGBM-only routing: OOMs 28 -> 1 at a modest wastage increase (`results/compare_old_vs_new.csv`).
- Active learning: gate vs random vs full-audit oracle across budgets 0.05-0.30 (`results/results_active_learning.csv`, `figures/fig_e3_al_curves`).
