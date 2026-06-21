#!/bin/bash
set -euo pipefail

AFTER_CSV="${1:?after csv path required}"
OUT_CSV="${2:?output csv path required}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ -f "$OUT_CSV" ]; then
  rm -f "$OUT_CSV" "${OUT_CSV}.lock"
fi

tail -n +2 "$AFTER_CSV" | while IFS=, read -r process task_hash m_cgroup_peak_bytes runtime_sec workdir; do
  bash "$SCRIPT_DIR/strace_replay.sh" "$workdir" "$process" "$task_hash" "$OUT_CSV"
done
