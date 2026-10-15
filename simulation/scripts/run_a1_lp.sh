#!/usr/bin/env bash
# A1-LP (gate G2a): run the fixed_repertoire and k_sweep arms of 03 A1.
#   scripts/run_a1_lp.sh <shard> <workers>
# Run one copy per host with a different shard index.  The paper arm and the
# k = 4 arm of the sweep are the same run-1 data (same RNG key), so neither is
# recomputed here.
set -euo pipefail
SHARD=$1; WORKERS=$2; NSH=${NSH:-3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONPATH=src
PY="nice -n 19 ionice -c 3 ./.venv/bin/python -m grhd.cli run"
mkdir -p logs

for n in 12 24 40; do
  $PY --design config/design_a1_fixed.yaml --n-rxn $n \
      --workers "$WORKERS" --shard "$SHARD" --n-shards "$NSH" \
      --chunk 50 --save-vectors none \
      --run-id a1lp_fixed_n${n}_s${SHARD}
done

for k in 2 8 16; do
  $PY --design config/design_a1_ksweep.yaml --k $k \
      --workers "$WORKERS" --shard "$SHARD" --n-shards "$NSH" \
      --chunk 50 --save-vectors none \
      --run-id a1lp_ksweep_k${k}_s${SHARD}
done
echo A1_LP_SHARD_${SHARD}_DONE
