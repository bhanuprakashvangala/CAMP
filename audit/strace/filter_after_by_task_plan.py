#!/usr/bin/env python3
"""Filter per_task_after_rerun.csv to only rows marked Audit in a task plan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


AUDIT_TRUE_VALUES = {"audit", "true", "yes", "y", "1"}


def normalize_process(value: str) -> str:
    return str(value or "").strip().replace(":", "-")


def normalize_hash(value: str) -> str:
    value = str(value or "").strip()
    if value.endswith("__dup2") or "__dup" in value:
        return value.split("__dup", 1)[0]
    return value


def should_audit(value: str) -> bool:
    return str(value or "").strip().lower() in AUDIT_TRUE_VALUES


def load_task_plan(path: str) -> set[tuple[str, str]]:
    audit_keys: set[tuple[str, str]] = set()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not should_audit(row.get("audit_flag", "")):
                continue
            task_type = normalize_process(row.get("task_type", ""))
            task_instance = normalize_hash(row.get("task_instance", ""))
            if task_type and task_instance:
                audit_keys.add((task_type, task_instance))
    return audit_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter per_task_after_rerun.csv by task_plan Audit rows.")
    parser.add_argument("--after", required=True, help="Input per_task_after_rerun.csv path.")
    parser.add_argument("--task-plan", required=True, help="Input task_plan.csv path.")
    parser.add_argument("--output", required=True, help="Output filtered after CSV path.")
    args = parser.parse_args()

    audit_keys = load_task_plan(args.task_plan)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    seen = 0
    with open(args.after, newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames or []
        with open(args.output, "w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                seen += 1
                key = (
                    normalize_process(row.get("process", "")),
                    normalize_hash(row.get("task_hash", "")),
                )
                if key not in audit_keys:
                    continue
                writer.writerow(row)
                kept += 1

    print(f"input_rows={seen}")
    print(f"audit_rows={len(audit_keys)}")
    print(f"kept_rows={kept}")
    print(f"output_csv={args.output}")


if __name__ == "__main__":
    main()
