#!/usr/bin/env python3
"""Load and validate selective audit task plans."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable


REQUIRED_COLUMNS = ("task_type", "task_instance", "predicted_memory", "audit_flag")
AUDIT_TRUE_VALUES = {"audit", "true", "yes", "y", "1"}


@dataclass(frozen=True)
class TaskPlanRow:
    task_type: str
    task_instance: str
    predicted_memory: str
    audit_flag: str
    row: dict[str, str]

    @property
    def key(self) -> tuple[str, str]:
        return self.task_type, self.task_instance

    @property
    def memory_mb(self) -> int:
        return parse_memory_mb(self.predicted_memory)

    @property
    def should_audit(self) -> bool:
        return normalize_audit_flag(self.audit_flag)


class TaskPlan:
    def __init__(self, rows: Iterable[TaskPlanRow]):
        self._rows: dict[tuple[str, str], TaskPlanRow] = {}
        for row in rows:
            if row.key in self._rows:
                task_type, task_instance = row.key
                raise ValueError(f"duplicate task plan row: {task_type}/{task_instance}")
            self._rows[row.key] = row

    def get_row(self, task_type: str, task_instance: str) -> TaskPlanRow:
        key = normalize_key(task_type), normalize_key(task_instance)
        try:
            return self._rows[key]
        except KeyError as exc:
            raise KeyError(f"missing task plan row: {task_type}/{task_instance}") from exc

    def get_memory_mb(self, task_type: str, task_instance: str) -> int:
        return self.get_row(task_type, task_instance).memory_mb

    def should_audit(self, task_type: str, task_instance: str) -> bool:
        return self.get_row(task_type, task_instance).should_audit

    def __len__(self) -> int:
        return len(self._rows)


def normalize_key(value: str) -> str:
    return str(value).strip()


def normalize_audit_flag(value: str) -> bool:
    return str(value).strip().lower() in AUDIT_TRUE_VALUES


def parse_memory_mb(value: str) -> int:
    text = str(value).strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)", text)
    if not match:
        raise ValueError(f"invalid predicted_memory value: {value!r}")

    number = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"", "m", "mb", "mib"}:
        multiplier = 1
    elif unit in {"g", "gb", "gib"}:
        multiplier = 1024
    elif unit in {"k", "kb", "kib"}:
        multiplier = 1 / 1024
    else:
        raise ValueError(f"unsupported memory unit in predicted_memory: {value!r}")

    memory_mb = math.ceil(number * multiplier)
    if memory_mb <= 0:
        raise ValueError(f"predicted_memory must be positive: {value!r}")
    return memory_mb


def detect_delimiter(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return ","
    if ext == ".tsv":
        return "\t"

    with open(path, newline="", encoding="utf-8") as handle:
        sample = handle.read(4096)
    if not sample.strip():
        raise ValueError(f"empty task plan: {path}")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        return "\t" if "\t" in sample.splitlines()[0] else ","


def load_task_plan(path: str) -> TaskPlan:
    delimiter = detect_delimiter(path)
    rows: list[TaskPlanRow] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"task plan has no header: {path}")

        fieldnames = [field.strip() for field in reader.fieldnames]
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"task plan missing required columns: {', '.join(missing)}")

        for line_number, raw_row in enumerate(reader, start=2):
            row = {str(key).strip(): (value or "").strip() for key, value in raw_row.items() if key}
            task_type = normalize_key(row["task_type"])
            task_instance = normalize_key(row["task_instance"])
            predicted_memory = row["predicted_memory"]
            audit_flag = row["audit_flag"]
            if not task_type or not task_instance:
                raise ValueError(f"empty task_type/task_instance at line {line_number}")
            parse_memory_mb(predicted_memory)
            rows.append(
                TaskPlanRow(
                    task_type=task_type,
                    task_instance=task_instance,
                    predicted_memory=predicted_memory,
                    audit_flag=audit_flag,
                    row=row,
                )
            )

    return TaskPlan(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and query a selective eBPF task plan.")
    parser.add_argument("--task-plan", required=True)
    parser.add_argument("--task-type")
    parser.add_argument("--task-instance")
    args = parser.parse_args()

    plan = load_task_plan(args.task_plan)
    if args.task_type or args.task_instance:
        if not args.task_type or not args.task_instance:
            parser.error("--task-type and --task-instance must be supplied together")
        row = plan.get_row(args.task_type, args.task_instance)
        print(f"task_type={row.task_type}")
        print(f"task_instance={row.task_instance}")
        print(f"memory_mb={row.memory_mb}")
        print(f"should_audit={str(row.should_audit).lower()}")
    else:
        print(f"valid_rows={len(plan)}")


if __name__ == "__main__":
    main()
