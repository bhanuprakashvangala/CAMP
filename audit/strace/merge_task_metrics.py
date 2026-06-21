#!/usr/bin/env python3
import argparse
import csv


def load_csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--after', required=True)
    ap.add_argument('--c-bytes', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    base_rows = load_csv(args.base)
    after_rows = {row['task_hash']: row for row in load_csv(args.after)}
    c_rows = {row['task_hash']: row for row in load_csv(args.c_bytes)}

    fieldnames = [
        'process',
        'task_hash',
        'a_bytes',
        'c_bytes',
        'M_peak_rss_bytes',
        'M_cgroup_peak_bytes',
        'runtime_seconds',
        'workdir',
    ]

    out_rows = []
    for row in base_rows:
        task_hash = row['task_hash']
        short_hash = task_hash[:8]
        c_row = c_rows.get(short_hash)
        if not c_row:
            continue
        after = after_rows.get(task_hash, {})
        out_rows.append({
            'process': row.get('process', ''),
            'task_hash': task_hash,
            'a_bytes': row.get('a_bytes', ''),
            'c_bytes': c_row.get('c_bytes', ''),
            'M_peak_rss_bytes': row.get('M_peak_rss_bytes', ''),
            'M_cgroup_peak_bytes': row.get('M_cgroup_peak_bytes', '') or after.get('m_cgroup_peak_bytes', ''),
            'runtime_seconds': row.get('runtime_seconds', '') or after.get('runtime_sec', ''),
            'workdir': after.get('workdir', '') or row.get('workdir', ''),
        })

    with open(args.output, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == '__main__':
    main()
