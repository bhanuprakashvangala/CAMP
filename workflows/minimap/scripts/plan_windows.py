#!/usr/bin/env python3
"""Create a deterministic manifest of synthetic Minimap2 benchmark windows."""

from __future__ import annotations

import argparse
import csv
import random


WINDOW_TYPES = {"unique_window": 0.40, "segdup_window": 0.35, "tr_window": 0.25}
PEAK_GB = {"unique_window": "8", "segdup_window": "12", "tr_window": "30+"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-windows", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7777)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    types = []
    for window_type, frac in WINDOW_TYPES.items():
        types.extend([window_type] * int(args.n_windows * frac))
    while len(types) < args.n_windows:
        types.append("tr_window")
    rng.shuffle(types)
    types = types[: args.n_windows]

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["window_id", "filename", "window_type", "expected_peak_gb", "seed"])
        for idx, window_type in enumerate(types, start=1):
            writer.writerow(
                [
                    idx,
                    f"window_{idx:04d}.fastq.gz",
                    window_type,
                    PEAK_GB[window_type],
                    args.seed + idx,
                ]
            )


if __name__ == "__main__":
    main()
