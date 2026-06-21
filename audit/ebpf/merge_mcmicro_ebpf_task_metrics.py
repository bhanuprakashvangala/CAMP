#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path


def parse_kv_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def parse_command_run_name(path: Path) -> str:
    pattern = re.compile(r"^### name:\s+'(.+)'$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    return ""


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
        name = parse_command_run_name(run_path) if run_path.exists() else ""
        trace_data = parse_kv_file(trace_path)
        mapping[task_hash] = {
            "work_dir": str(task_dir),
            "task_name": name,
            **trace_data,
        }
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-task eBPF outputs with Nextflow .command.trace metrics."
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Pipeline output directory containing ebpf/<PROCESS>/task_<HASH>/...",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Nextflow work directory for the same run.",
    )
    parser.add_argument(
        "--csv-out",
        required=True,
        help="Output CSV path for merged task metrics.",
    )
    parser.add_argument(
        "--tsv-out",
        default=None,
        help="Optional TSV companion output path for merged task metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    work_dir = Path(args.work_dir)
    csv_out = Path(args.csv_out)
    tsv_out = Path(args.tsv_out) if args.tsv_out else None

    work_map = find_work_task_map(work_dir)
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
        wrapper_pid = (
            wrapper_pid_path.read_text(encoding="utf-8").strip()
            if wrapper_pid_path.exists()
            else ""
        )

        trace_data = work_map.get(task_hash, {})
        row: dict[str, str | int] = {
            "process": process_name,
            "task_hash": task_hash,
            "task_name": trace_data.get("task_name", ""),
            "work_dir": trace_data.get("work_dir", ""),
            "wrapper_pid": wrapper_pid,
            "ebpf_tsv": str(ebpf_path),
            **ebpf_totals,
            "trace_realtime_ms": trace_data.get("realtime", ""),
            "trace_rchar": trace_data.get("rchar", ""),
            "trace_wchar": trace_data.get("wchar", ""),
            "trace_syscr": trace_data.get("syscr", ""),
            "trace_syscw": trace_data.get("syscw", ""),
            "trace_read_bytes": trace_data.get("read_bytes", ""),
            "trace_write_bytes": trace_data.get("write_bytes", ""),
            "trace_vmem_kb": trace_data.get("vmem", ""),
            "trace_rss_kb": trace_data.get("rss", ""),
            "trace_peak_vmem_kb": trace_data.get("peak_vmem", ""),
            "trace_peak_rss_kb": trace_data.get("peak_rss", ""),
            "trace_vol_ctxt": trace_data.get("vol_ctxt", ""),
            "trace_inv_ctxt": trace_data.get("inv_ctxt", ""),
        }
        rows.append(row)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "process",
        "task_hash",
        "task_name",
        "work_dir",
        "wrapper_pid",
        "ebpf_tsv",
        "ebpf_files",
        "ebpf_read_bytes",
        "ebpf_mmap_bytes",
        "ebpf_total_bytes",
        "trace_realtime_ms",
        "trace_rchar",
        "trace_wchar",
        "trace_syscr",
        "trace_syscw",
        "trace_read_bytes",
        "trace_write_bytes",
        "trace_vmem_kb",
        "trace_rss_kb",
        "trace_peak_vmem_kb",
        "trace_peak_rss_kb",
        "trace_vol_ctxt",
        "trace_inv_ctxt",
    ]

    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if tsv_out:
        tsv_out.parent.mkdir(parents=True, exist_ok=True)
        with tsv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(rows)} merged task rows to {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
