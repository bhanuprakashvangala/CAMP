#!/usr/bin/env python3
"""Run a workflow task normally or with PID-attached eBPF auditing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

from task_plan_loader import load_task_plan


def safe_part(value: str) -> str:
    return quote(value, safe="")


def metrics_path(metrics_dir: str, task_type: str, task_instance: str) -> Path:
    return Path(metrics_dir) / "tasks" / safe_part(task_type) / f"{safe_part(task_instance)}.json"


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def parse_ebpf_tsv(path: str) -> dict[str, int]:
    totals = {"ebpf_read_bytes": 0, "ebpf_mmap_bytes": 0, "ebpf_total_bytes": 0}
    if not path or not os.path.exists(path):
        return totals

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            totals["ebpf_read_bytes"] += int(row.get("read_bytes") or 0)
            totals["ebpf_mmap_bytes"] += int(row.get("mmap_bytes") or 0)
            totals["ebpf_total_bytes"] += int(row.get("total_bytes") or 0)
    return totals


def wait_for_ready_file(path: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return
        if proc.poll() is not None:
            raise RuntimeError(f"eBPF tracer exited before readiness with code {proc.returncode}")
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for eBPF ready file: {path}")


def stopped_child_preexec() -> None:
    os.kill(os.getpid(), signal.SIGSTOP)


def run_plain(command: list[str]) -> int:
    return subprocess.run(command).returncode


def run_audited(command: list[str], ebpf_script: str, ebpf_outdir: str, ready_timeout: float) -> tuple[int, int, str]:
    if os.geteuid() != 0:
        raise PermissionError("eBPF auditing requires root privileges")
    if not os.path.exists(ebpf_script):
        raise FileNotFoundError(f"eBPF script not found: {ebpf_script}")

    Path(ebpf_outdir).mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(command, preexec_fn=stopped_child_preexec)
    ready_file = os.path.join(ebpf_outdir, f"ready_{child.pid}.txt")
    if os.path.exists(ready_file):
        os.unlink(ready_file)

    tracer_stdout = open(os.path.join(ebpf_outdir, f"ebpf_audit_{child.pid}.stdout"), "w", encoding="utf-8")
    tracer_stderr = open(os.path.join(ebpf_outdir, f"ebpf_audit_{child.pid}.stderr"), "w", encoding="utf-8")
    tracer: subprocess.Popen | None = None
    try:
        tracer = subprocess.Popen(
            [
                sys.executable,
                ebpf_script,
                "--pid",
                str(child.pid),
                "--outdir",
                ebpf_outdir,
                "--ready-file",
                ready_file,
            ],
            stdout=tracer_stdout,
            stderr=tracer_stderr,
        )
        wait_for_ready_file(ready_file, tracer, ready_timeout)
        os.kill(child.pid, signal.SIGCONT)
        task_exit = child.wait()
        tracer_exit = tracer.wait()
        return task_exit, tracer_exit, os.path.join(ebpf_outdir, f"ebpf_attribution_{child.pid}.tsv")
    except Exception:
        if child.poll() is None:
            os.kill(child.pid, signal.SIGKILL)
            child.wait()
        if tracer is not None and tracer.poll() is None:
            tracer.terminate()
            tracer.wait(timeout=10)
        raise
    finally:
        tracer_stdout.close()
        tracer_stderr.close()


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run a task with optional eBPF auditing from a task plan.")
    parser.add_argument("--task-plan", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--task-instance", required=True)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--ebpf-script", default=str(script_dir / "ebpf_audit.py"))
    parser.add_argument("--ebpf-outdir", default=None)
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing task command after --")

    plan = load_task_plan(args.task_plan)
    plan_row = plan.get_row(args.task_type, args.task_instance)
    audit_requested = plan_row.should_audit
    ebpf_outdir = args.ebpf_outdir or os.path.join(args.metrics_dir, "ebpf")

    started = time.time()
    exit_code = 1
    tracer_exit_code: int | None = None
    ebpf_tsv_path = ""
    error = ""
    try:
        if audit_requested:
            exit_code, tracer_exit_code, ebpf_tsv_path = run_audited(
                command, args.ebpf_script, ebpf_outdir, args.ready_timeout
            )
        else:
            exit_code = run_plain(command)
    except Exception as exc:
        error = str(exc)
        exit_code = 1

    runtime_seconds = time.time() - started
    ebpf_totals = parse_ebpf_tsv(ebpf_tsv_path)
    audit_applied = audit_requested and not error and tracer_exit_code == 0

    if error:
        status = "wrapper_failed"
    elif exit_code == 0 and (not audit_requested or tracer_exit_code == 0):
        status = "success"
    elif audit_requested and tracer_exit_code not in (None, 0):
        status = "audit_failed"
    else:
        status = "task_failed"

    metric = {
        "task_type": args.task_type,
        "task_instance": args.task_instance,
        "predicted_memory_given": plan_row.memory_mb,
        "audit_flag": plan_row.audit_flag,
        "audit_requested": audit_requested,
        "audit_applied": audit_applied,
        "exit_code": exit_code,
        "tracer_exit_code": tracer_exit_code,
        "status": status,
        "runtime_seconds": round(runtime_seconds, 6),
        "ebpf_read_bytes": ebpf_totals["ebpf_read_bytes"],
        "ebpf_mmap_bytes": ebpf_totals["ebpf_mmap_bytes"],
        "ebpf_total_bytes": ebpf_totals["ebpf_total_bytes"],
        "ebpf_tsv_path": ebpf_tsv_path if audit_requested else "",
        "metrics_path": str(metrics_path(args.metrics_dir, args.task_type, args.task_instance)),
        "command": command,
        "error": error,
    }
    write_json_atomic(metrics_path(args.metrics_dir, args.task_type, args.task_instance), metric)

    if audit_requested and tracer_exit_code not in (None, 0) and exit_code == 0:
        sys.exit(70)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
