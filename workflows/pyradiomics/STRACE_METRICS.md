# Strace Metrics In This PyRadiomics Workflow

This workflow uses `strace` to measure syscall-level bytes for each Snakemake
task instance.

## What Strace Measures

`strace` observes system calls made by the task process and its child processes.
For byte accounting, this workflow traces read-like and write-like syscalls.

Read-like syscalls:

- `read`
- `pread64`
- `readv`
- `preadv`
- `preadv2`
- `recvfrom`
- `recvmsg`

Write-like syscalls:

- `write`
- `pwrite64`
- `writev`
- `pwritev`
- `pwritev2`
- `sendto`
- `sendmsg`

For each successful syscall, the workflow parses the return value after `=`.

Example:

```text
read(3, "...", 8192) = 4096
pread64(4, "...", 1024, 0) = 1024
write(1, "...", 200) = 200
```

The combined values are:

```text
strace_read_bytes_combined = 4096 + 1024 = 5120
strace_write_bytes_combined = 200
```

## Why Count More Than `read`

`read` alone is not enough. Programs and libraries can use different syscall
forms for different access patterns:

- `read`: basic sequential read.
- `pread64`: read at a specific file offset.
- `readv`: read into multiple buffers.
- `preadv`/`preadv2`: offset-based vector reads.
- `recvfrom`/`recvmsg`: socket reads.

These syscall categories do not overlap for the same event. A single syscall
event is either `read`, or `pread64`, or `readv`, and so on. Summing their
successful return values is not double-counting the same syscall.

One caveat: a program can read the same logical file bytes more than once. In
that case, strace counts both events because the task really requested those
bytes twice.

## Relationship To `rchar`

`rchar` from `/proc/<pid>/io` is conceptually similar to combined strace
read-like bytes:

```text
rchar ~= strace_read_bytes_combined
```

They can differ because:

- `rchar` is kernel-maintained per process.
- strace only counts the syscall names we trace and parse.
- child processes require `strace -f`.
- `/proc/<pid>/io` disappears when a process exits, so timing matters.
- both include more than normal file reads, such as pipes and sockets.

`read_bytes` is different. It estimates bytes actually fetched from storage and
can be much smaller than `rchar` when the filesystem cache is used.

## Output Files

Detailed metrics:

```text
results/task_metrics_detailed.csv
```

Compact Sizey-like metrics:

```text
results/task_metrics_sizey_like.csv
```

System configuration snapshot:

```text
results/system_config.json
```
