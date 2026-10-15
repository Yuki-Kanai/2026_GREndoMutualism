#!/usr/bin/env bash
# Post-evolution r = 1..100 (step 1, N_S = 300) for a queue of hosts, m = 10, k = 4.
#   scripts/posthost_queue.sh <slots> <workers_each> <hosts.txt>
# hosts.txt holds one host_id per line.
#
# Why a queue: worker counts are fixed at launch, so a fixed per-host allocation
# drains as hosts finish.  xargs -P refills a slot the moment a host is done.
#
# Each host lands in data/raw/post_paper{,_h<id>}, which already holds the A4 /
# run-1 blocks at N_S = 100 (host >= 1) or the full sweep (host 0).  Those are
# uniformly chunk 10, so this MUST use --chunk 10: a different block width would
# silently duplicate or drop symbionts (01_model_spec.md A25).
set -euo pipefail
SLOTS=$1; WORKERS=$2; SRC=${3:-/dev/stdin}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONPATH=src WORKERS
mkdir -p logs

run_host() {
  local H=$1
  nice -n 19 ionice -c 3 ./.venv/bin/python -m grhd.cli run \
    --design config/design_post_full.yaml --host-id "$H" \
    --chunk 10 --workers "$WORKERS" --save-vectors all \
    --run-id "postfullh${H}" > "logs/postfullh${H}.out" 2>&1
  echo "HOST_${H}_DONE"
}
export -f run_host

grep -v '^[[:space:]]*$' "$SRC" | xargs -P "$SLOTS" -L 1 bash -c 'run_host "$@"' _
echo "HOST_QUEUE_DONE"
