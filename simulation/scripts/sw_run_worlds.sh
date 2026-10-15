#!/usr/bin/env bash
# Shared-tournament check (SW) for one m: every world runs concurrently,
# pre (Level 1+2) then post (MILP) inside each world.
#   scripts/sw_run_worlds.sh <m> <workers_each> <post_design.yaml> <w> [<w> ...]
set -uo pipefail
M=$1; WE=$2; PD=$3; shift 3
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src
mkdir -p logs
TAG=sw_m${M}_$(hostname -s)
for W in "$@"; do
  (
    for ST in pre post; do
      D=config/design_sw_pre.yaml; [ $ST = post ] && D=$PD
      nice -n 19 ionice -c 3 ./.venv/bin/python -m grhd.cli run --design "$D" --m "$M" \
        --host-id "$W" --workers "$WE" --chunk 10 --run-id "sw_${ST}_m${M}_w${W}" \
        > logs/sw_${ST}_m${M}_w${W}.log 2>&1
      echo "SW_${ST}_M${M}_W${W}_DONE rc=$?" >> logs/${TAG}.log
    done
  ) &
done
wait
echo "SW_M${M}_ALL_DONE $*" >> logs/${TAG}.log
