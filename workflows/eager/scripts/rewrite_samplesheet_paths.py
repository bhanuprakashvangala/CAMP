#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def rewrite(value, data_dir):
    if value in ('', 'NA', 'na', 'Na', 'n/a', 'N/A'):
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
        reader = csv.DictReader(in_handle, delimiter='	')
        rows = list(reader)
        fieldnames = reader.fieldnames

    for row in rows:
        row['R1'] = rewrite(row['R1'], args.data_dir)
        row['R2'] = rewrite(row['R2'], args.data_dir)
        row['BAM'] = rewrite(row['BAM'], args.data_dir)

    with open(args.output, 'w', newline='') as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames, delimiter='	')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
