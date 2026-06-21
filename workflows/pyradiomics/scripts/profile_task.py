#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


READ_SYSCALLS = {
    "read",
    "pread64",
    "readv",
    "preadv",
    "preadv2",
    "recvfrom",
    "recvmsg",
}
WRITE_SYSCALLS = {
    "write",
    "pwrite64",
    "writev",
    "pwritev",
    "pwritev2",
    "sendto",
    "sendmsg",
}
SYSCALL_RE = re.compile(r"(?P<name>[A-Za-z0-9_]+)\(.*\)\s+=\s+(?P<ret>-?\d+)")


def stat_size(path: str) -> int:
    if not path or path == "-":
        return 0
    try:
        candidate = Path(path)
        if candidate.is_file():
            return candidate.stat().st_size
    except OSError:
        return 0
    return 0


def read_proc_io(pid: int) -> dict[str, int]:
    values = {
        "rchar": 0,
        "wchar": 0,
        "syscr": 0,
        "syscw": 0,
        "read_bytes": 0,
        "write_bytes": 0,
        "cancelled_write_bytes": 0,
    }
    try:
        with open(f"/proc/{pid}/io") as handle:
            for line in handle:
                key, raw_value = line.split(":", 1)
                if key in values:
                    values[key] = int(raw_value.strip())
    except OSError:
        pass
    return values


def read_rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def descendants_of(pid: int) -> list[int]:
    children: list[int] = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            close_paren = stat.rfind(")")
            fields = stat[close_paren + 2 :].split()
            ppid = int(fields[1])
        except (OSError, IndexError, ValueError):
            continue
        if ppid == pid:
            child_pid = int(entry.name)
            children.append(child_pid)
            children.extend(descendants_of(child_pid))
    return children


def aggregate_live_proc_io(root_pid: int) -> dict[str, int]:
    totals = {
        "rchar": 0,
        "wchar": 0,
        "syscr": 0,
        "syscw": 0,
        "read_bytes": 0,
        "write_bytes": 0,
        "cancelled_write_bytes": 0,
    }
    for pid in [root_pid, *descendants_of(root_pid)]:
        proc_io = read_proc_io(pid)
        for key in totals:
            totals[key] += proc_io[key]
    return totals


def aggregate_live_rss_kb(root_pid: int) -> int:
    return sum(read_rss_kb(pid) for pid in [root_pid, *descendants_of(root_pid)])


def parse_strace_log(path: Path) -> dict[str, int]:
    totals = {
        "strace_read_bytes_combined": 0,
        "strace_write_bytes_combined": 0,
        "strace_read_syscalls": 0,
        "strace_write_syscalls": 0,
        "strace_failed_syscalls": 0,
    }
    if not path.exists():
        return totals

    with path.open(errors="replace") as handle:
        for line in handle:
            match = SYSCALL_RE.search(line)
            if not match:
                continue
            syscall = match.group("name")
            returned = int(match.group("ret"))
            if returned < 0:
                totals["strace_failed_syscalls"] += 1
                continue
            if syscall in READ_SYSCALLS:
                totals["strace_read_bytes_combined"] += returned
                totals["strace_read_syscalls"] += 1
            elif syscall in WRITE_SYSCALLS:
                totals["strace_write_bytes_combined"] += returned
                totals["strace_write_syscalls"] += 1
    return totals


def system_config() -> dict[str, object]:
    cpu_count = os.cpu_count() or 0
    mem_total_kb = 0
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    mem_total_kb = int(line.split()[1])
                    break
    except OSError:
        pass
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": cpu_count,
        "mem_total_kb": mem_total_kb,
        "strace": shutil.which("strace") or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--sample", default="")
    parser.add_argument("--feature-class", default="")
    parser.add_argument("--declared-inputs", nargs="*", default=[])
    parser.add_argument("--original-inputs", nargs="*", default=[])
    parser.add_argument("--outputs", nargs="*", default=[])
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--strace-log", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("No command supplied to profile")

    metrics_path = Path(args.metrics_json)
    strace_path = Path(args.strace_log)
    stdout_path = Path(args.stdout_log)
    for path in [metrics_path, strace_path, stdout_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    strace_cmd = [
        "strace",
        "-f",
        "-yy",
        "-tt",
        "-T",
        "-e",
        "trace=read,pread64,readv,preadv,preadv2,recvfrom,recvmsg,write,pwrite64,writev,pwritev,pwritev2,sendto,sendmsg",
        "-o",
        str(strace_path),
        *command,
    ]

    original_input_size = sum(stat_size(path) for path in args.original_inputs)
    declared_input_size = sum(stat_size(path) for path in args.declared_inputs)
    start_time = time.time()
    start_proc_io: dict[str, int] = {}
    end_proc_io: dict[str, int] = {}
    max_proc_io = {
        "rchar": 0,
        "wchar": 0,
        "syscr": 0,
        "syscw": 0,
        "read_bytes": 0,
        "write_bytes": 0,
        "cancelled_write_bytes": 0,
    }
    peak_rss_kb = 0

    with stdout_path.open("w") as stdout_handle:
        process = subprocess.Popen(
            strace_cmd,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        start_proc_io = read_proc_io(process.pid)
        while process.poll() is None:
            live_io = aggregate_live_proc_io(process.pid)
            end_proc_io = live_io
            for key, value in live_io.items():
                max_proc_io[key] = max(max_proc_io[key], value)
            peak_rss_kb = max(peak_rss_kb, aggregate_live_rss_kb(process.pid))
            time.sleep(0.1)
        final_io = aggregate_live_proc_io(process.pid)
        if any(final_io.values()):
            end_proc_io = final_io
            for key, value in final_io.items():
                max_proc_io[key] = max(max_proc_io[key], value)
        elif any(max_proc_io.values()):
            end_proc_io = max_proc_io
        peak_rss_kb = max(peak_rss_kb, aggregate_live_rss_kb(process.pid))
        return_code = process.returncode

    end_time = time.time()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    runtime_seconds = end_time - start_time
    cpu_seconds = usage.ru_utime + usage.ru_stime
    cpu_utilization_percent = (
        (cpu_seconds / runtime_seconds) * 100 if runtime_seconds > 0 else 0
    )
    proc_delta = {
        key: max(0, end_proc_io.get(key, 0) - start_proc_io.get(key, 0))
        for key in [
            "rchar",
            "wchar",
            "syscr",
            "syscw",
            "read_bytes",
            "write_bytes",
            "cancelled_write_bytes",
        ]
    }
    strace_metrics = parse_strace_log(strace_path)
    output_size = sum(stat_size(path) for path in args.outputs)

    metrics = {
        "task_type": args.task_type,
        "task_id": args.task_id,
        "sample": args.sample,
        "feature_class": args.feature_class,
        "declared_inputs": args.declared_inputs,
        "original_inputs": args.original_inputs,
        "outputs": args.outputs,
        "input_file_size_bytes": original_input_size,
        "input_plus_intermediate_size_bytes": declared_input_size,
        "output_size_bytes": output_size,
        "runtime_seconds": runtime_seconds,
        "cpu_seconds": cpu_seconds,
        "cpu_utilization_percent": cpu_utilization_percent,
        "rss_kb": peak_rss_kb,
        "peak_rss_kb": max(peak_rss_kb, usage.ru_maxrss),
        "exit_code": return_code,
        "system_config": system_config(),
        **proc_delta,
        **strace_metrics,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
