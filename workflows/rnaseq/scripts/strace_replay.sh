#!/bin/bash
set -uo pipefail

WORK_DIR="${1:?work dir required}"
PROC="${2:?process required}"
HASH="${3:?hash required}"
OUT_CSV="${4:?output csv required}"

if [ ! -d "$WORK_DIR" ] || [ ! -f "$WORK_DIR/.command.run" ]; then
  echo "[strace_replay] skip: $WORK_DIR has no .command.run"
  exit 0
fi

PROC_TAG=$(echo "$PROC" | tr ':' '-' | tr ' ' '_' | tr -d '"')
HASH_TAG=$(echo "$HASH" | tr -d '/' | tr -d '"')
cd "$WORK_DIR"
STRACE_LOG="$WORK_DIR/.cstrace.txt"
STRACE_ERR="$WORK_DIR/.cstrace.err"
/usr/bin/strace -f \
    -e trace=read,pread64,readv,preadv \
    -o "$STRACE_LOG" 2>"$STRACE_ERR" \
    bash .command.run >/dev/null 2>&1

C_BYTES=0
if [ -s "$STRACE_LOG" ]; then
  C_BYTES=$(awk 'match($0, /read[v]?[0-9]*\(.*\) = ([0-9]+)/, m) {sum += m[1]} END {print sum+0}' "$STRACE_LOG")
fi

LOCK="${OUT_CSV}.lock"
(
  flock -x 9
  if [ ! -f "$OUT_CSV" ]; then
    echo "process,task_hash,c_bytes" > "$OUT_CSV"
  fi
  echo "${PROC_TAG},${HASH_TAG},${C_BYTES}" >> "$OUT_CSV"
) 9>"$LOCK"
