# Code Guide

This directory contains the ML-side scripts, preserved training outputs, and figure-generation code for the IEEE Cluster 2026 CAMP artifact.

## Layout

- `experiment_1/` contains the original point-prediction analysis scripts, figures, and paper-era notebook material.
- `experiment_2/` contains the probabilistic scoring and analysis scripts used for the eBPF lane.
- `experiment_3/` contains the active-learning evaluation scripts.
- `reruns/` contains the paper-faithful rerun path built around the shipped `per_task_full.pkl` model bundle.

## Reproduction

Treat this directory as the ML-side handoff after a workflow rerun has produced a workflow-level task table.

The sequence on the ML side is:

1. Reproduce a workflow from [../workflows/README.md](../workflows/README.md).
2. Build the workflow-level rerun table such as `task_metrics_rerun.csv` or the raw Experiment 2 eBPF rerun CSV.
3. Use the preserved prediction CSV for the corresponding experiment as the current gating input:
   - [../results/predictions_all.csv](../results/predictions_all.csv) for Experiment 1
   - [../results/predictions_qlgbm_exp2_all.csv](../results/predictions_qlgbm_exp2_all.csv) for Experiment 2
4. Run the gating step from [gating/README.md](gating/README.md).
5. Apply selective `strace` or selective `eBPF` only to the task instances marked `Audit`.
6. Merge the resulting `c_bytes` or eBPF audit features back into the workflow task table and compare against the preserved outputs.

This means the current artifact fully covers workflow rerun, gating, selective audit, and output validation. The reviewer does not need to regenerate the prediction CSVs in order to validate those paths.

## Preserved Inputs And Outputs

The main preserved ML-facing files used by the current artifact are:

- [../data/all_workflows.csv](../data/all_workflows.csv)
- [../data/task_metrics_pyradiomics_32k_detailed.csv](../data/task_metrics_pyradiomics_32k_detailed.csv)
- [../data/trace_methylseq.csv](../data/trace_methylseq.csv)
- [../data/ebpf](../data/ebpf)
- [../models/per_task_full.pkl](../models/per_task_full.pkl)
- [../models/models_exp2.pkl](../models/models_exp2.pkl)
- [../results/predictions_all.csv](../results/predictions_all.csv)
- [../results/predictions_qlgbm_exp2_all.csv](../results/predictions_qlgbm_exp2_all.csv)

## Directly Runnable Scripts

The scripts below are the main artifact-local entry points on the ML side:

```bash
# Train the shipped reruns point-prediction model family
python code/reruns/train_full.py

# Produce a deployment-time safe allocation for one task
python code/reruns/predict_memory_unified.py <workflow> <process> <a_bytes> [<c_bytes>]

# Reproduce the active-learning curves in the reruns path
python code/reruns/run_active_learning.py
```

The original `experiment_1/` and `experiment_2/` scripts are preserved as paper-era analysis code. Some of those scripts still assume the original internal training layout and are therefore best treated as provenance code unless you also recreate that earlier layout exactly.

## Current Scope Limitation

The current artifact fully supports:

- workflow rerun and workflow-output reconstruction
- gating reproduction
- selective `strace` reproduction for replay-based Experiment 1 workflows
- selective `eBPF` reproduction for Experiment 2 workflows
- validation against the preserved workflow and audit CSVs

This snapshot does not regenerate `results/predictions_all.csv` or `results/predictions_qlgbm_exp2_all.csv` directly from newly rerun workflow outputs; that bridging is out of scope for this snapshot. Use the preserved prediction CSVs listed above as the gating inputs for the workflow, gating, and selective-audit path.
