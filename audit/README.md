# Selective Auditing

CAMP recovers the consumption feature `c(t)` through two complementary
audit lanes. A task is audited only when the gate marks it `Audit`
(see [../code/gating/README.md](../code/gating/README.md)); everything
else runs uninstrumented.

- [`ebpf/`](ebpf) - online lane. A selective wrapper attaches an in-kernel
  eBPF tracer to a live task only when its plan row is marked `Audit`.
  Used by the Experiment 2 workflows (`bowtie`, `minimap`, `mcmicro`).
- [`strace/`](strace) - offline lane. Audited tasks are re-executed under
  `strace` to attribute consumed bytes, then merged back into the per-task
  table. Used by the Experiment 1 replay workflows.

Both lanes consume the same `task_plan.csv` (columns `task_type`,
`task_instance`, `predicted_memory`, `audit_flag`) produced by the gating
step.

The two workflows that run their own Nextflow process keep a local
`bin/merge_ebpf_trace.py` because `main.nf` resolves it via
`${workflow.projectDir}`; the shared tracer and wrapper live here.
