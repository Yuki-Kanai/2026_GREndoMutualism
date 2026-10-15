#!/usr/bin/env bash
# R5 / gate G9 -- like run_r5.sh, but runs several HOSTS concurrently.
#   scripts/run_r5_par.sh <design A|B> <m> <host_lo> <host_hi> <hosts_at_once> <workers_each> [N_S] [chunk] [low] [high]
#
# Why: run_r5.sh does one host at a time, and the last blocks of a host are the
# expensive ones (r = 40, 60, 100).  So every host ends with a long tail where
# only a handful of blocks are left and most workers sit idle, and the next host
# cannot start until that tail finishes.  Measured on cell2/3/6: 21-29 workers
# actually running out of 55-64 allocated.  Running several hosts at once
# overlaps the tails.
#
# `chunk` matters for small m, where a block of 10 MILPs takes milliseconds and
# the cost is dominated by writing two parquet files to the shared NFS home.
# Use 50 for m <= 14, 10 for the expensive m (more blocks = better load balance).
#
# Blocks are resumable, so this is safe to re-run over partly computed hosts.
set -euo pipefail
DESIGN=$1; M=$2; LO=$3; HI=$4; NPAR=$5; WORKERS=$6
NS=${7:-100}; CHUNK=${8:-10}; WLO=${9:-3}; WHI=${10:-6}
if [ "$DESIGN" = "A" ]; then
  RH=$(python3 -c "print(int(round(2.4*$M)))")
else
  RH=24
fi
GRID=$(python3 -c "
back={1,2,3,5,8,10,12,15,20,25,30,40,60,100}
fine=set(range(max(1,$M-1-$WLO), $M-1+$WHI+1))
print(','.join(str(x) for x in sorted(back|fine)))
")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONPATH=src
mkdir -p logs
echo "design=$DESIGN m=$M r_H=$RH npar=$NPAR workers=$WORKERS chunk=$CHUNK grid=$GRID"

run_one() {
  local H=$1
  local ARGS=(--design config/design_r5.yaml --host-id "$H" --r-grid "$GRID"
              --n-s "$NS" --workers "$WORKERS" --chunk "$CHUNK" --save-vectors all
              --run-id "r5${DESIGN}_m${M}_h${H}")
  if [ "$M" -ne 10 ] || [ "$RH" -ne 24 ]; then
    ARGS+=(--m "$M" --r-h "$RH")
  fi
  nice -n 19 ionice -c 3 ./.venv/bin/python -m grhd.cli run "${ARGS[@]}" \
    > "logs/r5${DESIGN}_m${M}_h${H}.out" 2>&1
  echo "R5_${DESIGN}_M${M}_HOST_${H}_DONE"
}

running=0
for H in $(seq "$LO" "$HI"); do
  run_one "$H" &
  running=$((running + 1))
  if [ "$running" -ge "$NPAR" ]; then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done
wait
echo "R5_${DESIGN}_M${M}_RANGE_${LO}_${HI}_DONE"
