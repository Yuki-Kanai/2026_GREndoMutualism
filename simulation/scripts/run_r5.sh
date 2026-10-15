#!/usr/bin/env bash
# R5 / gate G9 -- post-evolution MILP for one design and one m, over a host range.
#   scripts/run_r5.sh <design A|B> <m> <host_lo> <host_hi_inclusive> <workers> [N_S] [low] [high]
#
# r^H follows the design: A keeps r^H/m at the paper's m=10, r^H=24 ratio
# (r^H = round(2.4 m)); B fixes r^H = 24 (08_r5_two_designs.md section 0).
#
# The r grid is NOT the flat 15-point coarse grid.  r* is the output of R5, so its
# resolution is set by the grid, and the coarse grid only has 20, 25, 30 near the
# r ~ m-1 the paper predicts for m = 25.  So the grid is a backbone plus a step-1
# window around r = m-1.  r = 80 is dropped to pay for it; for m <= 15 that trade
# is a net saving, because cost grows with absolute r.
#
# The window is deliberately asymmetric, LOW below m-1 and HIGH above it (default
# -3 .. +6, so m = 25 gives r = 21..30).  The detlimit probe showed mu^H is
# underestimated by 1-5 % exactly in the r ~ 25 band at m >= 20, which pushes the
# measured r* BELOW the truth -- so the side that needs resolution is the upper one.
#
# At m = 10 with r^H = 24 the two designs coincide with the run-1 / A4 setting, so
# no --m / --r-h is passed: the run lands in data/raw/post_paper{,_h<id>} and the
# 15 blocks already computed there are skipped, leaving only the new r points.
set -euo pipefail
DESIGN=$1; M=$2; LO=$3; HI=$4; WORKERS=$5; NS=${6:-100}; WLO=${7:-3}; WHI=${8:-6}
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
echo "design=$DESIGN m=$M r_H=$RH grid=$GRID"
for H in $(seq "$LO" "$HI"); do
  ARGS=(--design config/design_r5.yaml --host-id "$H" --r-grid "$GRID"
        --n-s "$NS" --workers "$WORKERS" --chunk 10 --save-vectors all
        --run-id "r5${DESIGN}_m${M}_h${H}")
  if [ "$M" -ne 10 ] || [ "$RH" -ne 24 ]; then
    ARGS+=(--m "$M" --r-h "$RH")
  fi
  nice -n 19 ionice -c 3 ./.venv/bin/python -m grhd.cli run "${ARGS[@]}"
  echo "R5_${DESIGN}_M${M}_HOST_${H}_DONE"
done
echo "R5_${DESIGN}_M${M}_RANGE_${LO}_${HI}_DONE"
