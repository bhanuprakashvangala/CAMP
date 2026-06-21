#!/bin/bash
set -uo pipefail

PROC="${1:-unknown}"
HASH="${2:-unknown}"
SUMMARY_CSV="${3:?summary csv path required}"
PROC_TAG=$(echo "$PROC" | tr ':' '-' | tr ' ' '_' | tr -d '"')
HASH_TAG=$(echo "$HASH" | tr -d '/' | tr -d '"')
TASK_DIR="${PWD}"

MY_CG=$(awk -F: '{print $3}' /proc/self/cgroup 2>/dev/null | head -1)
M_CG_BYTES=0
if [ -n "$MY_CG" ] && [ -r "/sys/fs/cgroup${MY_CG}/memory.peak" ]; then
  M_CG_BYTES=$(cat "/sys/fs/cgroup${MY_CG}/memory.peak")
fi

START_TS=$(stat -c %Y "${TASK_DIR}/.command.begin" 2>/dev/null || echo 0)
END_TS=$(date +%s)
RUNTIME_SEC=$((END_TS - START_TS))
LOCK="${SUMMARY_CSV}.lock"
(
  flock -x 9
  if [ ! -f "$SUMMARY_CSV" ]; then
    echo "process,task_hash,m_cgroup_peak_bytes,runtime_sec,workdir" > "$SUMMARY_CSV"
  fi
  echo "${PROC_TAG},${HASH_TAG},${M_CG_BYTES},${RUNTIME_SEC},${TASK_DIR}" >> "$SUMMARY_CSV"
) 9>"$LOCK"
