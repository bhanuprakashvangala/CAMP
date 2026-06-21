#!/usr/bin/env python3
"""Generate a synthetic reference with segmental duplications and tandem repeats."""

from __future__ import annotations

import argparse
import os
import random


LINE = 60
BASES = list("ACGT")

TR_EXPANSIONS = [
    ("GGGGCC", 800, 10, "C9orf72_ALS"),
    ("CAG", 200, 20, "HTT_Huntington"),
    ("CTG", 800, 15, "DMPK_DM1"),
    ("GAA", 700, 8, "FXN_Friedreich"),
    ("CGG", 200, 30, "FMR1_FragileX"),
    ("AAGGG", 400, 5, "RFC1_CANVAS"),
    ("ATTCT", 600, 10, "ATXN10_SCA10"),
    ("TGGAA", 400, 8, "ATXN8_SCA8"),
]

CHROMS = [
    ("chr1", 1_200_000, 2, 2),
    ("chr2",   900_000, 2, 1),
    ("chr3",   700_000, 1, 1),
    ("chrX",   700_000, 1, 1),
]


def rseq(n: int, gc: float = 0.42) -> str:
    out = []
    for _ in range(n):
        r = random.random()
        if r < gc / 2:
            out.append("G")
        elif r < gc:
            out.append("C")
        elif r < gc + (1 - gc) / 2:
            out.append("A")
        else:
            out.append("T")
    return "".join(out)


def mutate(seq: str, rate: float) -> str:
    return "".join(
        random.choice([b for b in BASES if b != c]) if random.random() < rate else c for c in seq
    )


def build_chrom(name: str, length: int, n_segdups: int, n_tr: int) -> tuple[str, list[tuple]]:
    seq = list(rseq(length))
    anns = []
    used = []

    def place(content: str, label: str) -> bool:
        cl = len(content)
        if cl >= length - 1000:
            return False
        for _ in range(500):
            start = random.randint(1000, length - cl - 1000)
            if all(abs(start - u[0]) > cl + 500 for u in used):
                for i, c in enumerate(content):
                    seq[start + i] = c
                used.append((start, start + cl))
                anns.append((name, start, start + cl, label))
                return True
        return False

    for segdup_idx in range(n_segdups):
        plen = random.randint(8000, 15000)
        pseq = rseq(plen, gc=0.48)
        n_copies = random.randint(3, 6)
        place(pseq, f"segdup_{segdup_idx}_copy0")
        for copy_idx in range(1, n_copies):
            place(mutate(pseq, 0.03), f"segdup_{segdup_idx}_copy{copy_idx}")

    chosen = random.sample(TR_EXPANSIONS, min(n_tr, len(TR_EXPANSIONS)))
    for motif, n_exp, n_norm, disease in chosen:
        exp_seq = rseq(500) + motif * n_exp + rseq(500)
        norm_seq = rseq(500) + motif * n_norm + rseq(500)
        place(exp_seq, f"TR_expanded_{disease}")
        place(norm_seq, f"TR_normal_{disease}")

    return "".join(seq), anns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-fa", required=True)
    parser.add_argument("--out-bed", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_fa)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_bed)), exist_ok=True)

    all_anns = []
    with open(args.out_fa, "w", encoding="utf-8") as fa:
        for name, length, n_segdups, n_tr in CHROMS:
            seq, anns = build_chrom(name, length, n_segdups, n_tr)
            fa.write(f">{name}\n")
            for i in range(0, len(seq), LINE):
                fa.write(seq[i : i + LINE] + "\n")
            all_anns.extend(anns)

    with open(args.out_bed, "w", encoding="utf-8") as bed:
        bed.write("chrom\tstart\tend\tannotation\n")
        for row in sorted(all_anns):
            bed.write("\t".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
    main()
