#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def rewrite(value, data_dir):
    if not value:
        return value
    name = Path(value).name
    return str(Path(data_dir) / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    with open(args.input, newline='') as in_handle:
        reader = csv.DictReader(in_handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    for row in rows:
        row['short_reads_1'] = rewrite(row['short_reads_1'], args.data_dir)
        row['short_reads_2'] = rewrite(row['short_reads_2'], args.data_dir)
        row['long_reads'] = rewrite(row['long_reads'], args.data_dir)

    with open(args.output, 'w', newline='') as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
