#!/usr/bin/env python3
"""Rewrite the preserved MCMICRO sample sheet to point at a local data directory."""

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Preserved remote sample sheet CSV")
    parser.add_argument("--data-dir", required=True, help="Local directory holding the two OME-TIFF files")
    parser.add_argument("--output", required=True, help="Localized output sample sheet CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    with open(args.input, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = rows[0].keys() if rows else ["sample", "cycle_number", "image_tiles"]

    for row in rows:
        row["image_tiles"] = str(data_dir / Path(row["image_tiles"]).name)

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
