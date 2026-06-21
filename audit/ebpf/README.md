# eBPF audit lane (online)

This directory holds the shared eBPF tracing material used by the Experiment 2 workflows. See [../README.md](../README.md) for how it relates to the offline `strace` lane.

Files:
- `ebpf_audit.py`: shared tracer that attaches BCC probes to a live task PID and writes `ebpf_attribution_<PID>.tsv`
- `run_with_optional_audit.py`: shared selective wrapper that reads a task plan and attaches eBPF only for rows marked `Audit`
- `task_plan_loader.py`: validates and queries the gating task plan consumed by the selective wrapper
- `merge_mcmicro_ebpf_task_metrics.py`: merges `nf-core/mcmicro` per-task eBPF outputs with Nextflow `.command.trace` metrics

Requirements:
- run the instrumented workflow as `root`
- provide `python3-bpfcc`, `bpfcc-tools`, and matching `linux-headers-$(uname -r)`
- use Linux compute nodes; the tracing scripts are not Windows-native

Workflow usage:
- `bowtie` and `minimap` provide selective overlay configs that call `run_with_optional_audit.py` and consult the gating task plan before attaching eBPF
- `mcmicro` provides an artifact-local selective overlay config that uses `run_with_optional_audit.py`, `task_plan_loader.py`, and `ebpf_audit.py`

Sequence:
- run the baseline workflow command first, without the eBPF overlay
- generate the Experiment 2 gating task plan from the baseline prediction output
- rerun the same workflow with the selective eBPF overlay enabled and the generated `task_plan.csv`
