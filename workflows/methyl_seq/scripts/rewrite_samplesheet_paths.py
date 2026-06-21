#!/usr/bin/env python3
"""Rewrite methylseq samplesheet FASTQ paths to a reviewer-provided data dir."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rewrite_path(original: str, data_dir: Path) -> str:
    if not original:
        return original
    return str((data_dir / Path(original).name).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite fastq_1 and fastq_2 paths in a methylseq samplesheet so "
            "they point at a local reviewer data directory."
        )
    )
    parser.add_argument("--input", required=True, help="Source samplesheet CSV.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing the downloaded FASTQ files.",
    )
    parser.add_argument("--output", required=True, help="Rewritten samplesheet CSV.")
    args = parser.parse_args()

    input_path = Path(args.input)
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or ["sample", "fastq_1", "fastq_2", "genome"]
        rows = list(reader)

    for row in rows:
        row["fastq_1"] = rewrite_path(row.get("fastq_1", ""), data_dir)
        row["fastq_2"] = rewrite_path(row.get("fastq_2", ""), data_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
