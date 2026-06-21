#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as out_handle:
        writer = csv.writer(out_handle)
        writer.writerow(["sample", "feature_class", "feature", "value"])

        for input_file in args.inputs:
            with open(input_file, newline="") as in_handle:
                reader = csv.DictReader(in_handle)
                for row in reader:
                    writer.writerow(
                        [
                            args.sample,
                            row["feature_class"],
                            row["feature"],
                            row["value"],
                        ]
                    )


if __name__ == "__main__":
    main()
