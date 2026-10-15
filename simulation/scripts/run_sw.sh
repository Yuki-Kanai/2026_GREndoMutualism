#!/usr/bin/env bash
# Shared-tournament check (SW): pre then post for a list of worlds.
#   scripts/run_sw.sh <workers> <world_id> [<world_id> ...]
set -euo pipefail
WORKERS=$1; shift
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONPATH=src
mkdir -p logs
for STAGE in pre post; do
  for W in "$@"; do
    nice -n 19 ionice -c 3 ./.venv/bin/python -m grhd.cli run \
      --design config/design_sw_${STAGE}.yaml --host-id "$W" \
      --workers "$WORKERS" --chunk 10 --run-id "sw_${STAGE}_w${W}"
    echo "SW_${STAGE}_W${W}_DONE"
  done
done
echo "SW_ALL_DONE $*"
