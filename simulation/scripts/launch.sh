#!/usr/bin/env bash
# Launch one sharded run on one host, detached, low priority.
#   scripts/launch.sh <host> <shard> <workers> <design> <run-prefix> [extra args...]
set -euo pipefail
HOST=$1; SHARD=$2; WORKERS=$3; DESIGN=$4; PREFIX=$5; shift 5
RDIR='~/20260915_GRHD_Claude/20260917_GRHD_YKrepro'
ssh -n "$HOST" "cd $RDIR && mkdir -p logs && \
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
         NUMEXPR_NUM_THREADS=1 PYTHONPATH=src && \
  setsid nohup nice -n 19 ionice -c 3 ./.venv/bin/python -m grhd.cli run \
    --design $DESIGN --workers $WORKERS --shard $SHARD --n-shards 3 \
    --run-id ${PREFIX}_s${SHARD} $* \
    > logs/${PREFIX}_s${SHARD}.log 2>&1 < /dev/null & \
  sleep 4; echo \"$HOST shard $SHARD: \$(pgrep -cf grhd.cli) procs\""
