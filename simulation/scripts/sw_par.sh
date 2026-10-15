#!/usr/bin/env bash
# run several SW post worlds concurrently: sw_par.sh <workers_each> <w> [<w>...]
set -uo pipefail
WE=$1; shift
cd ~/20260915_GRHD_Claude/20260917_GRHD_YKrepro
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src
for W in "$@"; do
  ( nice -n 19 ionice -c 3 ./.venv/bin/python -m grhd.cli run --design config/design_sw_post.yaml \
      --host-id "$W" --workers "$WE" --chunk 10 --run-id "sw_post_w${W}" > logs/sw_post_w${W}.log 2>&1
    echo "SW_post_W${W}_DONE rc=$?" >> logs/sw_par_$(hostname -s).log ) &
done
wait
echo SW_PAR_DONE >> logs/sw_par_$(hostname -s).log
