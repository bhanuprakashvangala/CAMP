#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DETAILED_FIELDS = [
    "task_type",
    "task_id",
    "sample",
    "feature_class",
    "input_file_size_bytes",
    "input_plus_intermediate_size_bytes",
    "output_size_bytes",
    "rchar",
    "wchar",
    "syscr",
    "syscw",
    "read_bytes",
    "write_bytes",
    "cancelled_write_bytes",
    "strace_read_bytes_combined",
    "strace_write_bytes_combined",
    "strace_read_syscalls",
    "strace_write_syscalls",
    "strace_failed_syscalls",
    "rss_kb",
    "peak_rss_kb",
    "runtime_seconds",
    "cpu_seconds",
    "cpu_utilization_percent",
    "exit_code",
    "system_config_id",
    "declared_inputs",
    "original_inputs",
    "outputs",
]

SIZEY_FIELDS = [
    "task_type",
    "task_id",
    "sample",
    "feature_class",
    "input_file_size_bytes",
    "input_plus_intermediate_size_bytes",
    "rchar",
    "strace_read_bytes_combined",
    "read_bytes",
    "write_bytes",
    "wchar",
    "rss_kb",
    "peak_rss_kb",
    "runtime_seconds",
    "cpu_utilization_percent",
    "system_config_id",
]


def system_config_id(config: dict[str, object]) -> str:
    encoded = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def csv_value(value: object) -> object:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=[])
    parser.add_argument("--input-list")
    parser.add_argument("--detailed-csv", required=True)
    parser.add_argument("--sizey-csv", required=True)
    parser.add_argument("--system-config", required=True)
    args = parser.parse_args()

    input_paths = list(args.inputs)
    if args.input_list:
        input_paths.extend(
            line.strip()
            for line in Path(args.input_list).read_text().splitlines()
            if line.strip()
        )
    if not input_paths:
        raise ValueError("No metric JSON inputs supplied")

    rows = []
    configs: dict[str, dict[str, object]] = {}
    for input_path in input_paths:
        data = json.loads(Path(input_path).read_text())
        config = data.get("system_config", {})
        config_id = system_config_id(config)
        configs[config_id] = config
        data["system_config_id"] = config_id
        data.pop("system_config", None)
        rows.append(data)

    rows.sort(key=lambda row: (str(row["task_type"]), str(row["task_id"])))
    write_csv(Path(args.detailed_csv), rows, DETAILED_FIELDS)
    write_csv(Path(args.sizey_csv), rows, SIZEY_FIELDS)
    Path(args.system_config).write_text(json.dumps(configs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
