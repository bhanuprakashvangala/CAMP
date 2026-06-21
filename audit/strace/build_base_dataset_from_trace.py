#!/usr/bin/env python3
import argparse
import csv

UNITS = {
    'B': 1,
    'KB': 1024,
    'MB': 1024 ** 2,
    'GB': 1024 ** 3,
    'TB': 1024 ** 4,
}


def parse_size(value):
    value = (value or '').strip()
    if not value or value == '-':
        return ''
    parts = value.split()
    if len(parts) == 1:
        try:
            return str(int(float(parts[0].replace(',', ''))))
        except ValueError:
            return ''
    number = float(parts[0].replace(',', ''))
    unit = parts[1].upper()
    if unit not in UNITS:
        return ''
    return str(int(number * UNITS[unit]))


def derive_full_hash(workdir, short_hash):
    workdir = (workdir or '').strip()
    if workdir:
        parts = [p for p in workdir.replace('\\', '/').split('/') if p]
        if len(parts) >= 2:
            return parts[-2] + parts[-1]
    return short_hash.replace('/', '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    with open(args.trace, newline='') as handle:
        reader = csv.DictReader(handle, delimiter='	')
        rows = list(reader)

    out_rows = []
    for row in rows:
        task_hash = derive_full_hash(row.get('workdir', ''), row.get('hash', ''))
        out_rows.append({
            'process': row.get('process', ''),
            'task_hash': task_hash,
            'a_bytes': parse_size(row.get('rchar', '')),
            'c_bytes': '',
            'M_peak_rss_bytes': parse_size(row.get('peak_rss', '')),
            'M_cgroup_peak_bytes': '',
            'runtime_seconds': '',
            'workdir': row.get('workdir', ''),
        })

    fieldnames = ['process', 'task_hash', 'a_bytes', 'c_bytes', 'M_peak_rss_bytes', 'M_cgroup_peak_bytes', 'runtime_seconds', 'workdir']
    with open(args.output, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == '__main__':
    main()
