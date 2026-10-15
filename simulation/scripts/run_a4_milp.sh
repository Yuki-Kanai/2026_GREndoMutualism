#!/usr/bin/env bash
# A4 / gate G7 -- post-evolution MILP for a range of hosts.
#   scripts/run_a4_milp.sh <host_lo> <host_hi_inclusive> <workers>
# Host 0 writes into data/raw/post_paper, whose blocks already exist from run 1
# (the symbiont RNG key does not contain host_id), so it costs nothing and is
# reused verbatim.  Hosts >= 1 write into data/raw/post_paper_h<id>.
set -euo pipefail
LO=$1; HI=$2; WORKERS=$3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONPATH=src
mkdir -p logs
for H in $(seq "$LO" "$HI"); do
  nice -n 19 ionice -c 3 ./.venv/bin/python -m grhd.cli run \
    --design config/design_a4.yaml --host-id "$H" \
    --workers "$WORKERS" --chunk 10 --save-vectors all \
    --run-id "a4_h${H}"
  echo "A4_HOST_${H}_DONE"
done
echo "A4_RANGE_${LO}_${HI}_DONE"
