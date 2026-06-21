#!/usr/bin/env python3
"""Merge per-task eBPF outputs with Nextflow trace rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_command_run_name(path: Path) -> str:
    pattern = re.compile(r"^### name:\s+'(.+)'$")
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    return ""


def parse_kv_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def prefer_trace_row(current: dict[str, str] | None, candidate: dict[str, str]) -> dict[str, str]:
    if current is None:
        return candidate

    rank = {
        "COMPLETED": 4,
        "CACHED": 3,
        "RUNNING": 2,
        "SUBMITTED": 1,
    }

    current_rank = rank.get(current.get("status", ""), 0)
    candidate_rank = rank.get(candidate.get("status", ""), 0)
    if candidate_rank != current_rank:
        return candidate if candidate_rank > current_rank else current

    try:
        current_attempt = int(current.get("attempt", "0") or 0)
    except ValueError:
        current_attempt = 0
    try:
        candidate_attempt = int(candidate.get("attempt", "0") or 0)
    except ValueError:
        candidate_attempt = 0

    return candidate if candidate_attempt >= current_attempt else current


def aggregate_ebpf(tsv_path: Path) -> dict[str, int]:
    totals = {
        "ebpf_files": 0,
        "ebpf_read_bytes": 0,
        "ebpf_mmap_bytes": 0,
        "ebpf_total_bytes": 0,
    }
    with tsv_path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            totals["ebpf_files"] += 1
            totals["ebpf_read_bytes"] += int(row.get("read_bytes", "0") or 0)
            totals["ebpf_mmap_bytes"] += int(row.get("mmap_bytes", "0") or 0)
            totals["ebpf_total_bytes"] += int(row.get("total_bytes", "0") or 0)
    return totals


def find_work_task_map(work_dir: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for trace_path in work_dir.glob("*/*/.command.trace"):
        task_dir = trace_path.parent
        task_hash = f"{task_dir.parent.name}{task_dir.name}"
        run_path = task_dir / ".command.run"
        name = parse_command_run_name(run_path)
        trace_data = parse_kv_file(trace_path)
        mapping[task_hash] = {
            "work_dir": str(task_dir),
            "task_name": name,
            **trace_data,
        }
    return mapping


def read_trace_maps(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_workdir: dict[str, dict[str, str]] = {}
    by_name: dict[str, dict[str, str]] = {}
    if not path.exists():
        return by_workdir, by_name
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            workdir = row.get("workdir", "").strip()
            if workdir:
                by_workdir[workdir] = prefer_trace_row(by_workdir.get(workdir), row)
            name = row.get("name", "").strip()
            if name:
                by_name[name] = prefer_trace_row(by_name.get(name), row)
    return by_workdir, by_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Nextflow trace and per-task eBPF outputs.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--tsv-out", required=True)
    return parser.parse_args()


def write_table(path: Path, rows: list[dict], fieldnames: list[str], delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    work_dir = Path(args.work_dir)
    trace_path = Path(args.trace)

    work_map = find_work_task_map(work_dir)
    trace_by_workdir, trace_by_name = read_trace_maps(trace_path)
    rows: list[dict[str, str | int]] = []

    for task_dir in sorted(outdir.glob("ebpf/*/task_*")):
        process_name = task_dir.parent.name
        task_hash = task_dir.name.removeprefix("task_")
        ebpf_files = sorted(task_dir.glob("ebpf_attribution_*.tsv"))
        if not ebpf_files:
            continue

        ebpf_path = ebpf_files[0]
        ebpf_totals = aggregate_ebpf(ebpf_path)
        wrapper_pid_path = task_dir / "wrapper_pid.txt"
        launch_pid_path = task_dir / "task_pid.txt"
        audit_pid_path = task_dir / "audit_launcher_pid.txt"
        wrapper_pid = wrapper_pid_path.read_text(encoding="utf-8").strip() if wrapper_pid_path.exists() else ""
        launch_pid = launch_pid_path.read_text(encoding="utf-8").strip() if launch_pid_path.exists() else ""
        audit_pid = audit_pid_path.read_text(encoding="utf-8").strip() if audit_pid_path.exists() else ""

        command_trace_data = work_map.get(task_hash, {})
        work_dir = command_trace_data.get("work_dir", "")
        trace_data = trace_by_workdir.get(work_dir, {})
        if not trace_data:
            trace_data = trace_by_name.get(command_trace_data.get("task_name", ""), {})

        row: dict[str, str | int] = {
            "process": process_name,
            "task_hash": task_hash,
            "task_name": command_trace_data.get("task_name", ""),
            "work_dir": work_dir,
            "wrapper_pid": wrapper_pid,
            "launch_pid": launch_pid,
            "audit_pid": audit_pid,
            "ebpf_tsv": str(ebpf_path),
            **ebpf_totals,
            "trace_task_id": trace_data.get("task_id", ""),
            "trace_native_id": trace_data.get("native_id", ""),
            "trace_tag": trace_data.get("tag", ""),
            "trace_status": trace_data.get("status", ""),
            "trace_exit": trace_data.get("exit", ""),
            "trace_attempt": trace_data.get("attempt", ""),
            "trace_cpus": trace_data.get("cpus", ""),
            "trace_time": trace_data.get("time", ""),
            "trace_memory": trace_data.get("memory", ""),
            "trace_duration": trace_data.get("duration", ""),
            "trace_realtime": trace_data.get("realtime", ""),
            "trace_queue": trace_data.get("queue", ""),
            "trace_cpu_pct": trace_data.get("%cpu", ""),
            "trace_mem_pct": trace_data.get("%mem", ""),
            "trace_rchar": trace_data.get("rchar", ""),
            "trace_wchar": trace_data.get("wchar", ""),
            "trace_syscr": trace_data.get("syscr", ""),
            "trace_syscw": trace_data.get("syscw", ""),
            "trace_read_bytes": trace_data.get("read_bytes", ""),
            "trace_write_bytes": trace_data.get("write_bytes", ""),
            "trace_vmem": trace_data.get("vmem", ""),
            "trace_rss": trace_data.get("rss", ""),
            "trace_peak_vmem": trace_data.get("peak_vmem", ""),
            "trace_peak_rss": trace_data.get("peak_rss", ""),
        }
        rows.append(row)

    fieldnames = [
        "process",
        "task_hash",
        "task_name",
        "work_dir",
        "wrapper_pid",
        "launch_pid",
        "audit_pid",
        "ebpf_tsv",
        "ebpf_files",
        "ebpf_read_bytes",
        "ebpf_mmap_bytes",
        "ebpf_total_bytes",
        "trace_task_id",
        "trace_native_id",
        "trace_tag",
        "trace_status",
        "trace_exit",
        "trace_attempt",
        "trace_cpus",
        "trace_time",
        "trace_memory",
        "trace_duration",
        "trace_realtime",
        "trace_queue",
        "trace_cpu_pct",
        "trace_mem_pct",
        "trace_rchar",
        "trace_wchar",
        "trace_syscr",
        "trace_syscw",
        "trace_read_bytes",
        "trace_write_bytes",
        "trace_vmem",
        "trace_rss",
        "trace_peak_vmem",
        "trace_peak_rss",
    ]

    write_table(Path(args.csv_out), rows, fieldnames, ",")
    write_table(Path(args.tsv_out), rows, fieldnames, "\t")
    print(f"Wrote {len(rows)} merged task rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
