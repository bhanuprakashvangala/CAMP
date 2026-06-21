# PyRadiomics TotalSegmentator Run: End-to-End Summary

Historical provenance note:
- This file preserves the original cluster-era run notes and `/shared/...` paths from the recorded run history.
- For artifact review and fresh reproduction from this repository, use the artifact-local commands in [../README.md](../README.md) instead of the commands or absolute paths recorded here.

## Scope

This document summarizes the complete PyRadiomics benchmark run carried out in:

- `/shared/pyradiomics/workflow_totalseg_32k_strict`

The goal was to build a real, validated, multi-step radiomics workflow over TotalSegmentator CT images and ROI masks, execute it on the cluster, and produce task-level metrics suitable for downstream modeling and benchmarking.

## Dataset Source

The run used CT volumes and organ segmentation masks from:

- `/shared/pyradiomics/workflow_datasets/totalsegmentator_v201`

Five ROI classes were used per case:

1. `liver`
2. `spleen`
3. `kidney_left`
4. `kidney_right`
5. `pancreas`

## Initial Candidate Pool

The initial usable candidate sheet for the 5-ROI workflow contained:

- `3887` ROI samples

This pool had already excluded obviously unusable inputs such as unreadable CTs and empty masks from earlier preprocessing.

## Strict Pre-Screening

Before the final large run, the candidate pool was re-screened using stricter PyRadiomics-compatible validation logic. The purpose was to eliminate ROI/image pairs that would fail during feature extraction even if the files existed.

The strict screen removed:

- unreadable or invalid inputs
- ROI masks rejected by PyRadiomics mask validation
- degenerate masks such as single-voxel ROIs

Strict screening results:

- strict-valid ROI samples: `3882`
- strict-invalid ROI samples removed: `5`

Relevant files:

- valid subset used for the run: [usable_samples_32k_strict.tsv](/shared/pyradiomics/workflow_totalseg_32k_strict/usable_samples_32k_strict.tsv)
- invalid strict-screened rows: [strict_invalid_samples.tsv](/shared/pyradiomics/workflow_totalseg_32k_rerun/strict_invalid_samples.tsv)

Example rejected reason:

- `mask only contains 1 segmented voxel! Cannot extract features for a single voxel`

## Final Run Sheet

From the strict-valid pool, a clean run sheet was prepared with:

- `3556` ROI samples

File:

- [usable_samples_32k_strict.tsv](/shared/pyradiomics/workflow_totalseg_32k_strict/usable_samples_32k_strict.tsv)

This was chosen to produce a task-metrics dataset of about `32k` task instances.

## Workflow Structure

Configuration:

- [config_32k.yaml](/shared/pyradiomics/workflow_totalseg_32k_strict/config_32k.yaml)
- [Params_32k.yaml](/shared/pyradiomics/workflow_totalseg_32k_strict/Params_32k.yaml)
- [Snakefile](/shared/pyradiomics/workflow_totalseg_32k_strict/Snakefile)

Feature classes extracted per ROI:

1. `firstorder`
2. `shape`
3. `glcm`
4. `glrlm`
5. `glszm`
6. `gldm`
7. `ngtdm`

Per ROI sample, the workflow executed:

1. `validate_inputs`
2. `extract_feature_class` for each of the 7 feature classes
3. `merge_sample_features`

Global workflow steps:

1. `merge_all_samples`
2. `aggregate_task_metrics`

## Expected Job Counts

For `3556` ROI samples:

- `3556` validation tasks
- `24892` feature-class extraction tasks
- `3556` per-sample merge tasks
- `1` global merge task
- `1` final metrics aggregation task
- `1` Snakemake `all` target

Total Snakemake DAG size:

- `32007` jobs

Task-instance count represented in the final task metrics CSVs:

- `32005` data rows

This is:

- `3556 + 24892 + 3556 + 1 = 32005`

The `aggregate_task_metrics` rule itself and the `all` target are not represented as separate data rows in the final task-metrics CSVs.

## Instrumentation and Metrics

Each workflow task was profiled through the workflow scripts under:

- `/shared/pyradiomics/workflow_totalseg_32k_strict/scripts`

The task profiling captured:

- declared input sizes
- original input sizes
- runtime
- RSS
- peak RSS
- CPU metrics
- `rchar`
- `wchar`
- `read_bytes`
- `write_bytes`
- syscall-level `strace` read/write information

Final outputs include:

- detailed task metrics CSV
- sizey-style task metrics CSV
- system configuration JSON

## Resource Settings

Extraction tasks were configured with:

- `1` CPU per task
- `2000 MB` memory per extraction task
- `240` minute extraction runtime limit

From configuration:

- `extract_mem_mb: 2000`
- `extract_runtime_min: 240`

The run used cluster execution through Slurm with many single-core tasks in parallel.

## Execution Environment

System configuration was recorded in:

- [system_config.json](/shared/pyradiomics/workflow_totalseg_32k_strict/results/system_config.json)

Observed execution environment summary:

- execution hosts observed: `19`
- host families used:
  - `compute-dy-nodes-1` through `compute-dy-nodes-18`
  - `compute-st-nodes-1`
- CPUs per execution host: `32`
- memory per execution host: about `261130xxx KB` (`~249 GiB`)
- Python version: `3.10.12`
- kernel: `6.8.0-1050-aws`
- `strace`: `/usr/bin/strace`

## Run Behavior and Final Fix

The core workflow execution completed successfully:

- all validation tasks completed
- all feature extraction tasks completed
- all per-sample merge tasks completed
- the global merged feature matrix was produced successfully

The only late-stage failure was the final `aggregate_task_metrics` Slurm job, which was first submitted with an insufficient wall-clock limit and was cancelled due to time limit.

This did not affect feature extraction outputs. It only delayed creation of the final aggregated metrics tables.

The final aggregation was then completed afterward, producing the final metrics files.

## Final Outputs

Feature matrix:

- [all_features.csv](/shared/pyradiomics/workflow_totalseg_32k_strict/results/all_features.csv)

Task metrics:

- [task_metrics_detailed.csv](/shared/pyradiomics/workflow_totalseg_32k_strict/results/task_metrics_detailed.csv)
- [task_metrics_sizey_like.csv](/shared/pyradiomics/workflow_totalseg_32k_strict/results/task_metrics_sizey_like.csv)
- [system_config.json](/shared/pyradiomics/workflow_totalseg_32k_strict/results/system_config.json)

Row counts:

- `all_features.csv`: `380493` lines total
  - `380492` data rows plus header
- `task_metrics_detailed.csv`: `32006` lines total
  - `32005` data rows plus header
- `task_metrics_sizey_like.csv`: `32006` lines total
  - `32005` data rows plus header

Output sizes at the time of summary:

- `all_features.csv`: about `27 MB`
- `task_metrics_detailed.csv`: about `22 MB`
- `task_metrics_sizey_like.csv`: about `5.6 MB`
- `system_config.json`: about `9.9 KB`

## Reproducibility Notes

The clean strict run directory contains the final runnable state:

- [config_32k.yaml](/shared/pyradiomics/workflow_totalseg_32k_strict/config_32k.yaml)
- [Params_32k.yaml](/shared/pyradiomics/workflow_totalseg_32k_strict/Params_32k.yaml)
- [Snakefile](/shared/pyradiomics/workflow_totalseg_32k_strict/Snakefile)
- [usable_samples_32k_strict.tsv](/shared/pyradiomics/workflow_totalseg_32k_strict/usable_samples_32k_strict.tsv)

This run therefore produced:

1. a fully validated ROI-based PyRadiomics workflow
2. a real cluster execution over `3556` ROI samples
3. `24892` radiomics extraction task instances
4. `32005` task-level metric rows
5. a merged feature matrix with `380492` data rows

## Short Method Summary

For paper text, the run can be described succinctly as follows:

> We constructed a PyRadiomics workflow over TotalSegmentator CT volumes using five organ ROIs per case (`liver`, `spleen`, `kidney_left`, `kidney_right`, `pancreas`). An initial 5-ROI candidate pool of 3887 ROI samples was re-screened with strict PyRadiomics-compatible validation, yielding 3882 valid ROI samples and excluding five invalid masks. From this pool, 3556 ROI samples were selected for a large cluster run. Each ROI underwent input validation, seven PyRadiomics feature-class extraction tasks, and per-sample feature merging, followed by global feature merging and task-metrics aggregation. The resulting workflow executed 32007 Snakemake jobs in total and produced 24892 extraction task instances and 32005 task-level metric rows. Final outputs included a merged feature matrix (`380492` data rows), detailed task metrics, a sizey-style metrics table, and system configuration metadata.
