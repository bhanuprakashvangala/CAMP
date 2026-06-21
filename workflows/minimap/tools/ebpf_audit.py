#!/usr/bin/env python3
# 08_ebpf_audit.py
#
# Attach eBPF probes to vfs_read and filemap_fault to record per-file byte
# attribution for a target workflow task.
#
# For a given PID or wrapped command, it records:
#   - How many bytes each open file contributed via read() syscalls (vfs_read)
#   - How many bytes each open file contributed via mmap page faults
#     (filemap_fault)
#
# The sum of these two per file gives per-file data consumption during
# execution — the feature that /proc/<pid>/io read_bytes cannot provide
# because it only exposes an aggregate process total.
#
# Requires:
#   sudo apt-get install -y bpfcc-tools python3-bpfcc linux-headers-$(uname -r)
#
# Usage (attach to a running task):
#   sudo python3 08_ebpf_audit.py --pid <PID>
#
# Usage (wrap a command, audit from start):
#   sudo python3 08_ebpf_audit.py -- python3 workflow_task.py ...
#
# Output:
#   A TSV of (filename, read_bytes, mmap_bytes, total_bytes) written to
#   <outdir>/ebpf_attribution_<PID>.tsv and a human-readable summary printed to
#   stderr.

import argparse
import ctypes
import os
import subprocess
import sys
import time

BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>
#include <linux/mm.h>
#include <linux/sched.h>

struct file_bytes_t {
    u64 read_bytes;
    u64 mmap_bytes;
};

struct filename_t {
    char name[128];
};

struct file_key_t {
    u32 tgid;
    char name[128];
};

struct fd_key_t {
    u32 tgid;
    int fd;
};

struct pending_fd_t {
    int fd;
};

BPF_HASH(tracked_pids, u32, u8, 32768);
BPF_HASH(file_bytes, struct file_key_t, struct file_bytes_t, 10240);
BPF_HASH(pending_open, u64, struct filename_t, 32768);
BPF_HASH(fd_to_name, struct fd_key_t, struct filename_t, 32768);
BPF_HASH(pending_read, u64, struct pending_fd_t, 32768);
BPF_PERCPU_ARRAY(scratch_file_key, struct file_key_t, 1);
BPF_HASH(debug_counts, u32, u64, 64);
BPF_HASH(seen_tgids, u32, u64, 1024);

TRACEPOINT_PROBE(sched, sched_process_fork)
{
    u32 parent_pid = args->parent_pid;
    u32 child_pid = args->child_pid;

    u8 *tracked = tracked_pids.lookup(&parent_pid);
    if (!tracked)
        return 0;

    u8 one = 1;
    tracked_pids.update(&child_pid, &one);
    return 0;
}

static __always_inline int is_tracked_pid(u32 pid)
{
#ifdef DISABLE_TRACK_FILTER
    return 1;
#else
    u8 *tracked = tracked_pids.lookup(&pid);
    if (!tracked)
        return 0;
    return 1;
#endif
}

static __always_inline void bump_counter(u32 key)
{
    u64 zero = 0;
    u64 *val = debug_counts.lookup_or_try_init(&key, &zero);
    if (val) {
        __sync_fetch_and_add(val, 1);
    }
}

static __always_inline void note_seen_pid(u32 pid)
{
    u64 zero = 0;
    u64 *val = seen_tgids.lookup_or_try_init(&pid, &zero);
    if (val) {
        __sync_fetch_and_add(val, 1);
    }
}

static __always_inline int record_read_bytes(u32 tgid, int fd, ssize_t ret)
{
    if (ret <= 0)
        return 0;

    struct fd_key_t fd_key = {};
    fd_key.tgid = tgid;
    fd_key.fd = fd;

    struct filename_t *fname = fd_to_name.lookup(&fd_key);
    if (!fname) {
        bump_counter(11);
        return 0;
    }

    u32 zero_idx = 0;
    struct file_key_t *key = scratch_file_key.lookup(&zero_idx);
    if (!key)
        return 0;

    __builtin_memset(key, 0, sizeof(*key));
    key->tgid = tgid;
    __builtin_memcpy(key->name, fname->name, sizeof(key->name));

    struct file_bytes_t zero = {};
    struct file_bytes_t *val = file_bytes.lookup_or_try_init(key, &zero);
    if (val) {
        __sync_fetch_and_add(&val->read_bytes, ret);
        bump_counter(12);
    }
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_openat)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    bump_counter(31);
    u32 pid = pid_tgid;
    u32 tgid = pid_tgid >> 32;
    note_seen_pid(pid);
    if (!is_tracked_pid(pid))
        return 0;

    bump_counter(1);
    struct filename_t fname = {};
    bpf_probe_read_user_str(fname.name, sizeof(fname.name), args->filename);
    pending_open.update(&pid_tgid, &fname);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_openat)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid;
    u32 tgid = pid_tgid >> 32;
    if (!is_tracked_pid(pid))
        return 0;

    bump_counter(2);
    int fd = args->ret;
    struct filename_t *fname = pending_open.lookup(&pid_tgid);
    if (!fname)
        return 0;

    if (fd >= 0) {
        struct fd_key_t fd_key = {};
        fd_key.tgid = tgid;
        fd_key.fd = fd;
        fd_to_name.update(&fd_key, fname);
    }

    pending_open.delete(&pid_tgid);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_openat2)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid;
    u32 tgid = pid_tgid >> 32;
    if (!is_tracked_pid(pid))
        return 0;

    bump_counter(3);
    struct filename_t fname = {};
    bpf_probe_read_user_str(fname.name, sizeof(fname.name), args->filename);
    pending_open.update(&pid_tgid, &fname);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_openat2)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid;
    u32 tgid = pid_tgid >> 32;
    if (!is_tracked_pid(pid))
        return 0;

    bump_counter(4);
    int fd = args->ret;
    struct filename_t *fname = pending_open.lookup(&pid_tgid);
    if (!fname)
        return 0;

    if (fd >= 0) {
        struct fd_key_t fd_key = {};
        fd_key.tgid = tgid;
        fd_key.fd = fd;
        fd_to_name.update(&fd_key, fname);
    }

    pending_open.delete(&pid_tgid);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_read)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    bump_counter(32);
    u32 pid = pid_tgid;
    u32 tgid = pid_tgid >> 32;
    note_seen_pid(pid);
    if (!is_tracked_pid(pid))
        return 0;

    bump_counter(5);
    struct pending_fd_t pending = {};
    pending.fd = args->fd;
    pending_read.update(&pid_tgid, &pending);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_read)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid;
    u32 tgid = pid_tgid >> 32;
    if (!is_tracked_pid(pid))
        return 0;

    bump_counter(6);
    struct pending_fd_t *pending = pending_read.lookup(&pid_tgid);
    if (!pending)
        return 0;

    record_read_bytes(tgid, pending->fd, args->ret);
    pending_read.delete(&pid_tgid);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_pread64)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid;
    u32 tgid = pid_tgid >> 32;
    if (!is_tracked_pid(pid))
        return 0;

    bump_counter(7);
    struct pending_fd_t pending = {};
    pending.fd = args->fd;
    pending_read.update(&pid_tgid, &pending);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_pread64)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid;
    u32 tgid = pid_tgid >> 32;
    if (!is_tracked_pid(pid))
        return 0;

    bump_counter(8);
    struct pending_fd_t *pending = pending_read.lookup(&pid_tgid);
    if (!pending)
        return 0;

    record_read_bytes(tgid, pending->fd, args->ret);
    pending_read.delete(&pid_tgid);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_enter_close)
{
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid;
    u32 tgid = pid_tgid >> 32;
    if (!is_tracked_pid(pid))
        return 0;

    bump_counter(9);
    struct fd_key_t fd_key = {};
    fd_key.tgid = tgid;
    fd_key.fd = args->fd;
    fd_to_name.delete(&fd_key);
    return 0;
}

int trace_filemap_fault(struct pt_regs *ctx, struct vm_fault *vmf)
{
    if (!vmf || !vmf->vma)
        return 0;

    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid;
    u32 tgid = pid_tgid >> 32;
    if (!is_tracked_pid(pid))
        return 0;

    bump_counter(10);
    struct file *file = vmf->vma->vm_file;
    if (!file)
        return 0;

    struct dentry *de = file->f_path.dentry;
    if (!de)
        return 0;

    u32 zero_idx = 0;
    struct file_key_t *key = scratch_file_key.lookup(&zero_idx);
    if (!key)
        return 0;

    __builtin_memset(key, 0, sizeof(*key));
    key->tgid = tgid;
    bpf_probe_read_kernel_str(key->name, sizeof(key->name), de->d_name.name);

    struct file_bytes_t zero = {};
    struct file_bytes_t *val = file_bytes.lookup_or_try_init(key, &zero);
    if (val) {
        __sync_fetch_and_add(&val->mmap_bytes, 4096);
    }
    return 0;
}
"""


def attach_and_collect(
    target_pid: int,
    duration: float | None,
    outdir: str,
    proc: subprocess.Popen | None = None,
    no_filter: bool = False,
    ready_file: str | None = None,
) -> None:
    try:
        from bcc import BPF
    except ImportError:
        print("ERROR: python-bcc is not installed.", file=sys.stderr)
        print(
            "  sudo apt-get install -y bpfcc-tools python3-bpfcc "
            "linux-headers-$(uname -r)",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        root_tgid = target_pid
        os.kill(root_tgid, 0)
    except ProcessLookupError:
        print(f"ERROR: PID {target_pid} does not exist.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Attaching eBPF probes to root TGID={root_tgid} (PID={target_pid}) and descendants",
        file=sys.stderr,
    )

    program_text = BPF_PROGRAM
    if no_filter:
        program_text = "#define DISABLE_TRACK_FILTER 1\n" + program_text

    b = BPF(text=program_text)
    tracked_pids = b.get_table("tracked_pids")
    if not no_filter:
        tracked_pids[ctypes.c_uint(root_tgid)] = ctypes.c_ubyte(1)
    b.attach_kprobe(event="filemap_fault", fn_name="trace_filemap_fault")
    if ready_file:
        ready_dir = os.path.dirname(ready_file)
        if ready_dir:
            os.makedirs(ready_dir, exist_ok=True)
        with open(ready_file, "w", encoding="utf-8") as handle:
            handle.write(f"{target_pid}\n")

    start = time.time()
    try:
        while True:
            if proc is not None:
                if proc.poll() is not None:
                    break
            else:
                try:
                    os.kill(target_pid, 0)
                except ProcessLookupError:
                    break

            if duration is not None and (time.time() - start) >= duration:
                break

            time.sleep(0.2)
    finally:
        file_bytes_map = b.get_table("file_bytes")
        results: dict[str, dict[str, int | str]] = {}
        for k, v in file_bytes_map.items():
            raw_name = bytes(k.name).rstrip(b"\x00").decode("utf-8", errors="replace")
            if raw_name not in results:
                results[raw_name] = {
                    "filename": raw_name or "<unknown>",
                    "read_bytes": 0,
                    "mmap_bytes": 0,
                }
            results[raw_name]["read_bytes"] += v.read_bytes
            results[raw_name]["mmap_bytes"] += v.mmap_bytes

        rows = sorted(
            results.values(),
            key=lambda r: int(r["read_bytes"]) + int(r["mmap_bytes"]),
            reverse=True,
        )

        print("\n" + "=" * 80, file=sys.stderr)
        print(
            f"{'Filename':<40}  {'read() bytes':>14}  "
            f"{'mmap bytes':>14}  {'total':>14}",
            file=sys.stderr,
        )
        print("-" * 80, file=sys.stderr)
        for r in rows:
            total = int(r["read_bytes"]) + int(r["mmap_bytes"])
            print(
                f"{str(r['filename']):<40}  {int(r['read_bytes']):>14,}  "
                f"{int(r['mmap_bytes']):>14,}  {total:>14,}",
                file=sys.stderr,
            )
        print("=" * 80, file=sys.stderr)

        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, f"ebpf_attribution_{target_pid}.tsv")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("filename\tread_bytes\tmmap_bytes\ttotal_bytes\n")
            for r in rows:
                total = int(r["read_bytes"]) + int(r["mmap_bytes"])
                f.write(
                    f"{r['filename']}\t{int(r['read_bytes'])}\t"
                    f"{int(r['mmap_bytes'])}\t{total}\n"
                )
        print(f"\nAttribution written to: {out_path}", file=sys.stderr)

        debug_counts_map = b.get_table("debug_counts")
        debug_labels = {
            1: "enter_openat",
            2: "exit_openat",
            3: "enter_openat2",
            4: "exit_openat2",
            5: "enter_read",
            6: "exit_read",
            7: "enter_pread64",
            8: "exit_pread64",
            9: "enter_close",
            10: "filemap_fault",
            11: "read_missing_fd_name",
            12: "read_bytes_recorded",
            31: "prefilter_enter_openat",
            32: "prefilter_enter_read",
        }
        if debug_counts_map:
            print("\nDebug counters:", file=sys.stderr)
            for key in sorted(debug_counts_map.keys(), key=lambda x: x.value):
                label = debug_labels.get(key.value, str(key.value))
                print(f"  {label}: {debug_counts_map[key].value}", file=sys.stderr)

        seen_tgids_map = b.get_table("seen_tgids")
        if seen_tgids_map:
            print("\nSeen PIDs:", file=sys.stderr)
            for key in sorted(seen_tgids_map.keys(), key=lambda x: x.value):
                print(f"  pid={key.value}: {seen_tgids_map[key].value}", file=sys.stderr)

        b.detach_kprobe(event="filemap_fault")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="eBPF per-file I/O attribution for workflow tasks"
    )
    parser.add_argument("--pid", type=int, default=None, help="PID of running process to audit")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Seconds to collect (default: until process exits)",
    )
    parser.add_argument(
        "--outdir",
        default="metrics",
        help="Directory for TSV output (default: metrics/)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable TGID/descendant filtering and trace all file activity on the node during the command lifetime.",
    )
    parser.add_argument("--ready-file", default=None, help="Optional file to create once probes are attached")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to launch and audit (alternative to --pid)")
    args = parser.parse_args()
    proc: subprocess.Popen | None = None

    if args.pid is not None and args.cmd:
        parser.error("Specify either --pid or a command, not both.")

    if args.pid is not None:
        target_pid = args.pid
    elif args.cmd:
        if args.cmd[0] == "--":
            args.cmd = args.cmd[1:]
        if not args.cmd:
            parser.error("No command given after --")
        proc = subprocess.Popen(args.cmd)
        target_pid = proc.pid
    else:
        parser.error("Specify --pid <PID> or a command to run.")

    if os.geteuid() != 0:
        print("ERROR: eBPF requires root. Run with sudo.", file=sys.stderr)
        sys.exit(1)

    attach_and_collect(
        target_pid,
        args.duration,
        args.outdir,
        proc=proc if args.cmd else None,
        no_filter=args.no_filter,
        ready_file=args.ready_file,
    )


if __name__ == "__main__":
    main()
