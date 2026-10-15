#!/usr/bin/env bash
# A1-MILP (gate G2b, the main verdict).
#   scripts/run_a1_milp.sh <shard> <workers> <n_rxn>
# Only the fixed_repertoire arm is computed: the paper arm of the same grid,
# same N_S and the same host is already in data/raw/post_paper (identical RNG
# key), so re-running it would produce byte-identical output.
set -euo pipefail
SHARD=$1; WORKERS=$2; NRXN=$3; NSH=${NSH:-3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONPATH=src
mkdir -p logs
nice -n 19 ionice -c 3 ./.venv/bin/python -m grhd.cli run \
  --design config/design_a1_post.yaml --n-rxn "$NRXN" \
  --workers "$WORKERS" --shard "$SHARD" --n-shards "$NSH" \
  --chunk 25 --save-vectors all \
  --run-id a1milp_n${NRXN}_s${SHARD}
echo A1_MILP_SHARD_${SHARD}_DONE
