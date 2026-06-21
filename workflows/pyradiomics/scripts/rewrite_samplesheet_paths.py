#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


TOKEN = "__TOTALSEG_DATA_DIR__"


def rewrite_path(value: str, data_dir: Path) -> str:
    return value.replace(TOKEN, data_dir.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    data_dir = Path(args.data_dir).resolve()

    with input_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        for row in rows:
            row["image"] = rewrite_path(row["image"], data_dir)
            row["mask"] = rewrite_path(row["mask"], data_dir)
            writer.writerow(row)


if __name__ == "__main__":
    main()
