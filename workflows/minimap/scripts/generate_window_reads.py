#!/usr/bin/env python3
"""Generate one synthetic long-read FASTQ window deterministically."""

from __future__ import annotations

import argparse
import gzip
import random


BASES = list("ACGT")
COMP = str.maketrans("ACGT", "TGCA")

READ_PARAMS = {
    "unique_window": {"mean": 2500, "sd": 800, "min": 1500, "max": 5000, "n": 8},
    "segdup_window": {"mean": 4000, "sd": 1200, "min": 2000, "max": 7000, "n": 6},
    "tr_window": {"mean": 8000, "sd": 2500, "min": 4000, "max": 15000, "n": 2},
}


def load_genome(path: str) -> dict[str, str]:
    genome = {}
    name = None
    buf = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip()
            if line.startswith(">"):
                if name:
                    genome[name] = "".join(buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if name:
        genome[name] = "".join(buf)
    return genome


def load_annotations(path: str) -> dict[str, list[tuple[str, int, int]]]:
    anns = {"segdup": [], "TR_expanded": []}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("chrom"):
                continue
            chrom, start, end, annotation = line.rstrip().split("\t")
            for key in anns:
                if annotation.startswith(key):
                    anns[key].append((chrom, int(start), int(end)))
    return anns


def ont_error(seq: str, rng: random.Random, rate: float = 0.08) -> str:
    out = []
    i = 0
    while i < len(seq):
        r = rng.random()
        if r < rate / 3:
            out.append(rng.choice(BASES))
            i += 1
        elif r < 2 * rate / 3:
            out.append(rng.choice(BASES))
        elif r < rate:
            i += 1
        else:
            out.append(seq[i])
            i += 1
    return "".join(out)


def sample_read(
    genome: dict[str, str],
    chrom: str,
    region_start: int,
    region_end: int,
    target_len: int,
    rng: random.Random,
) -> str | None:
    seq = genome.get(chrom, "")
    actual = min(target_len, region_end - region_start)
    if actual < 1000 or region_end - region_start < 1:
        return None
    frag_start = max(0, min(rng.randint(region_start, max(region_start, region_end - actual)), len(seq) - actual - 1))
    frag = seq[frag_start : frag_start + actual]
    frag = "".join(base if base in set(BASES) else rng.choice(BASES) for base in frag)
    if rng.random() < 0.5:
        frag = frag[::-1].translate(COMP)
    return ont_error(frag, rng)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genome", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--window-id", type=int, required=True)
    parser.add_argument("--window-type", required=True, choices=sorted(READ_PARAMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    genome = load_genome(args.genome)
    anns = load_annotations(args.annotations)
    params = READ_PARAMS[args.window_type]

    if args.window_type == "unique_window":
        chrom = rng.choice(list(genome.keys()))
        regions = [
            (
                chrom,
                rng.randint(1000, len(genome[chrom]) // 2),
                rng.randint(len(genome[chrom]) // 2, len(genome[chrom]) - 1000),
            )
        ]
    elif args.window_type == "segdup_window":
        regions = anns.get("segdup", [])
        if not regions:
            chrom = list(genome.keys())[0]
            regions = [(chrom, 10000, 100000)]
    else:
        regions = anns.get("TR_expanded", [])
        if not regions:
            chrom = list(genome.keys())[0]
            regions = [(chrom, 10000, 500000)]

    with gzip.open(args.out, "wt", encoding="utf-8") as handle:
        written = 0
        for _ in range(params["n"]):
            read_len = int(rng.gauss(params["mean"], params["sd"]))
            read_len = max(params["min"], min(params["max"], read_len))
            chrom, region_start, region_end = rng.choice(regions)
            if args.window_type == "tr_window":
                region_start = max(0, region_start - 5000)
                region_end = min(len(genome.get(chrom, "")) - 1, region_end + 5000)
            read = sample_read(genome, chrom, region_start, region_end, read_len, rng)
            if not read or len(read) < 1000:
                continue
            qual = "0" * len(read)
            written += 1
            handle.write(
                f"@w{args.window_id}_r{written} type={args.window_type} len={len(read)}\n"
                f"{read}\n+\n{qual}\n"
            )


if __name__ == "__main__":
    main()
