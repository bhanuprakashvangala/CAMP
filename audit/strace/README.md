# strace audit lane (offline)

Shared scripts for the replay-based Experiment 1 workflows. Audited tasks
are re-executed under `strace` to recover the consumed-byte feature `c(t)`,
then merged back into the per-task table.

- `filter_after_by_task_plan.py` - keep only the `Audit` rows of a rerun
  per-task table, using the gating `task_plan.csv`.
- `replay_c_bytes_from_after_csv.sh` - drive the replay over the filtered
  table; calls `strace_replay.sh` (kept alongside it) per task.
- `strace_replay.sh` - re-run one task workdir under `strace` and emit its
  consumed bytes.
- `after_task.sh` - Nextflow `afterScript` hook that records the per-task
  workdir and runtime needed for replay.
- `build_base_dataset_from_trace.py` - build the base per-task table from a
  recorded Nextflow trace.
- `merge_task_metrics.py` - merge recorded metrics with replayed `c_bytes`.

These are referenced from the per-workflow commands in
[../../workflows/README.md](../../workflows/README.md). `replay_c_bytes`
locates `strace_replay.sh` from its own directory, so keep the two files
together.
