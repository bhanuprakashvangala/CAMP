#!/usr/bin/env python3
"""
Re-join the per-workflow data with the c values that the original run.py join missed.

run.py builds the dataset CSV but loses c_bytes due to a hash-format mismatch:
  - nf-trace.txt        hash field    = "xx/yyyyyy" prefix form
  - per_task_after.csv  task_hash     = full 32-char hash
  - per_task_c.csv      task_hash     = first 8 chars of the full hash

Common key: extract full hash from nf-trace.txt's workdir column,
then match c rows by first 8 chars.

Output: one CSV per workflow with c_bytes populated, plus a unified all_workflows.csv.
"""
from __future__ import annotations

import csv, subprocess, shlex, sys
from pathlib import Path

HEAD = "ubuntu@100.30.185.54"
KEY = "/c/Users/govin/.ssh/mempred-slurm-key.pem"
SSH_OPTS = ["-i", KEY, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new"]

WORKFLOWS = ["methylseq", "chipseq", "rnaseq", "eager", "mag", "mag_karlsson"]

LOCAL_OUT = Path(__file__).parent / "output" / "joined"
LOCAL_OUT.mkdir(parents=True, exist_ok=True)


def fetch_remote(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["scp", *SSH_OPTS, f"{HEAD}:{remote}", str(local)],
        capture_output=True, text=True
    )
    return res.returncode == 0


def to_int_bytes(s: str) -> str:
    s = (s or "").strip()
    if not s: return ""
    if s.endswith(" KB"): return str(int(float(s[:-3]) * 1024))
    if s.endswith(" MB"): return str(int(float(s[:-3]) * 1024 * 1024))
    if s.endswith(" GB"): return str(int(float(s[:-3]) * 1024 * 1024 * 1024))
    if s.endswith(" B"):  return str(int(float(s[:-2])))
    try: return str(int(s))
    except: return ""


def full_hash_from_workdir(wd: str) -> str:
    parts = wd.rstrip("/").split("/")
    if len(parts) >= 2:
        return parts[-2] + parts[-1]
    return ""


def join_workflow(workflow: str) -> dict:
    """Pull the 3 source files for a workflow and produce a joined CSV."""
    print(f"\n=== {workflow} ===")
    wd = LOCAL_OUT / workflow
    wd.mkdir(exist_ok=True)
    trace_p = wd / "nf-trace.txt"
    after_p = wd / "per_task_after.csv"
    c_p     = wd / "per_task_c.csv"

    # fetch
    for rname, lpath in [
        (f"/shared/{workflow}_run/nf-trace.txt", trace_p),
        (f"/shared/{workflow}_run/per_task_after.csv", after_p),
        (f"/shared/{workflow}_run/per_task_c.csv", c_p),
    ]:
        ok = fetch_remote(rname, lpath)
        size = lpath.stat().st_size if lpath.exists() else 0
        print(f"  {'OK' if ok else 'MISS':4} {rname} ({size} bytes)")

    # parse nf-trace
    nf_by_full_hash = {}
    if trace_p.exists() and trace_p.stat().st_size > 0:
        with open(trace_p) as f:
            rdr = csv.reader(f, delimiter="\t")
            head = next(rdr, None)
            if head:
                for row in rdr:
                    if len(row) < len(head): continue
                    d = dict(zip(head, row))
                    if d.get("status") != "COMPLETED": continue
                    h = full_hash_from_workdir(d.get("workdir", ""))
                    if h:
                        nf_by_full_hash[h] = d

    # parse after (full hash key). per_task_after.csv may contain a SECOND
    # row per hash from the strace c-replay (it re-executes .command.run, whose
    # exit trap re-fires Nextflow's afterScript). The first row is the original
    # workflow execution with the genuine M/runtime; the second is a stub
    # (M~few MB, runtime~0). Keep the first occurrence.
    after_by_full = {}
    if after_p.exists() and after_p.stat().st_size > 0:
        with open(after_p) as f:
            for r in csv.DictReader(f):
                after_by_full.setdefault(r["task_hash"], r)

    # parse c (8-char short key)
    c_by_short = {}
    if c_p.exists() and c_p.stat().st_size > 0:
        with open(c_p) as f:
            for r in csv.DictReader(f):
                c_by_short[r["task_hash"]] = r["c_bytes"]

    # write joined
    out_p = wd / f"{workflow}_dataset.csv"
    n_rows = 0
    n_with_c = 0
    with open(out_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workflow", "process", "task_hash",
                    "a_bytes", "c_bytes",
                    "M_peak_rss_bytes", "M_cgroup_peak_bytes", "runtime_seconds"])
        for full_h, d in nf_by_full_hash.items():
            short_h = full_h[:8]
            af = after_by_full.get(full_h, {})
            c = c_by_short.get(short_h, "")
            if c: n_with_c += 1
            w.writerow([
                workflow,
                d.get("process", ""),
                full_h,
                to_int_bytes(d.get("rchar", "")),
                c,
                to_int_bytes(d.get("peak_rss", "")),
                af.get("m_cgroup_peak_bytes", ""),
                af.get("runtime_sec", ""),
            ])
            n_rows += 1
    print(f"  joined dataset: {n_rows} rows ({n_with_c} with c) -> {out_p}")
    return {"workflow": workflow, "rows": n_rows, "with_c": n_with_c, "path": out_p}


def import_sizey_iwd() -> dict:
    """iwd is from sizey's trace_iwd.csv; we don't have workdirs to replay.
    Map a=rchar, M=peak_rss, runtime=realtime/1000, c=NaN."""
    src = Path(__file__).parent.parent / "sizey" / "data" / "trace_iwd.csv"
    out_p = LOCAL_OUT / "iwd" / "iwd_dataset.csv"
    if not src.exists():
        return {"workflow": "iwd", "rows": 0, "with_c": 0, "path": out_p}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(src) as fin, open(out_p, "w", newline="") as fout:
        rdr = csv.DictReader(fin)
        wout = csv.writer(fout)
        wout.writerow(["workflow", "process", "task_hash", "a_bytes", "c_bytes",
                       "M_peak_rss_bytes", "M_cgroup_peak_bytes", "runtime_seconds"])
        for r in rdr:
            if r.get("status") != "COMPLETED": continue
            try:
                a = int(r.get("rchar") or 0)
                m = int(r.get("peak_rss") or 0)
                rt = int(r.get("realtime") or 0) / 1000.0
            except ValueError:
                continue
            if a <= 0 or m <= 0: continue
            wout.writerow(["iwd", r.get("process",""),
                           (r.get("hash","")).replace("/", ""),
                           a, "", m, m, rt])
            n_rows += 1
    print(f"=== iwd (from sizey) ===")
    print(f"  joined dataset: {n_rows} rows (0 with c) -> {out_p}")
    return {"workflow": "iwd", "rows": n_rows, "with_c": 0, "path": out_p}


def main():
    summaries = []
    for wf in WORKFLOWS:
        summaries.append(join_workflow(wf))
    summaries.append(import_sizey_iwd())

    # combine all into one CSV
    all_p = LOCAL_OUT / "all_workflows.csv"
    n_total = 0
    n_with_c = 0
    with open(all_p, "w", newline="") as fout:
        wout = csv.writer(fout)
        wrote_header = False
        for s in summaries:
            if not s["path"].exists(): continue
            with open(s["path"]) as fin:
                rdr = csv.reader(fin)
                head = next(rdr)
                if not wrote_header:
                    wout.writerow(head); wrote_header = True
                for row in rdr:
                    wout.writerow(row); n_total += 1
                    if row[4]: n_with_c += 1   # c_bytes column

    print(f"\n=== combined dataset: {all_p} ===")
    print(f"  total rows: {n_total}")
    print(f"  with c_bytes: {n_with_c} ({n_with_c*100//max(n_total,1)}%)")
    print(f"\nper-workflow summary:")
    for s in summaries:
        print(f"  {s['workflow']:10} {s['rows']:>4} rows, {s['with_c']:>4} with c")


if __name__ == "__main__":
    main()
