#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split paired gzipped FASTQ files into numbered chunks.")
    parser.add_argument("--run-accession", required=True)
    parser.add_argument("--fastq1", required=True)
    parser.add_argument("--fastq2", required=True)
    parser.add_argument("--chunks", required=True, type=int)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def count_reads(path: Path) -> int:
    lines = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for _ in handle:
            lines += 1
    return lines // 4


def read_record(handle) -> list[str] | None:
    lines = [handle.readline() for _ in range(4)]
    if not lines[0]:
        return None
    if any(line == "" for line in lines):
        raise RuntimeError("Truncated FASTQ record encountered during split.")
    return lines


def main() -> int:
    args = parse_args()
    fq1 = Path(args.fastq1)
    fq2 = Path(args.fastq2)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    total_reads = count_reads(fq1)
    chunks = max(1, args.chunks)
    per_chunk = max(1, (total_reads + chunks - 1) // chunks)

    rows: list[dict[str, str | int]] = []
    current_chunk = 1
    current_count = 0
    chunk_reads = 0
    h1 = h2 = None
    chunk1 = chunk2 = None

    def open_chunk(chunk_id: int):
        c1 = outdir / f"{args.run_accession}.chunk_{chunk_id:03d}_1.fastq.gz"
        c2 = outdir / f"{args.run_accession}.chunk_{chunk_id:03d}_2.fastq.gz"
        return (
            c1,
            c2,
            gzip.open(c1, "wt", encoding="utf-8"),
            gzip.open(c2, "wt", encoding="utf-8"),
        )

    try:
        chunk1, chunk2, h1, h2 = open_chunk(current_chunk)
        with gzip.open(fq1, "rt", encoding="utf-8", errors="replace") as r1, \
             gzip.open(fq2, "rt", encoding="utf-8", errors="replace") as r2:
            while True:
                rec1 = read_record(r1)
                rec2 = read_record(r2)
                if rec1 is None and rec2 is None:
                    break
                if rec1 is None or rec2 is None:
                    raise RuntimeError("FASTQ pair length mismatch encountered during split.")

                if current_count >= per_chunk and current_chunk < chunks:
                    h1.close()
                    h2.close()
                    rows.append({
                        "run_accession": args.run_accession,
                        "chunk_id": current_chunk,
                        "fastq1": str(chunk1.resolve()),
                        "fastq2": str(chunk2.resolve()),
                        "n_pairs": chunk_reads,
                    })
                    current_chunk += 1
                    current_count = 0
                    chunk_reads = 0
                    chunk1, chunk2, h1, h2 = open_chunk(current_chunk)

                h1.writelines(rec1)
                h2.writelines(rec2)
                current_count += 1
                chunk_reads += 1
    finally:
        if h1:
            h1.close()
        if h2:
            h2.close()

    rows.append({
        "run_accession": args.run_accession,
        "chunk_id": current_chunk,
        "fastq1": str(chunk1.resolve()),
        "fastq2": str(chunk2.resolve()),
        "n_pairs": chunk_reads,
    })

    manifest = Path(args.manifest)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_accession", "chunk_id", "fastq1", "fastq2", "n_pairs"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} chunks for {args.run_accession}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
