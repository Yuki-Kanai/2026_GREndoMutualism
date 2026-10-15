#!/usr/bin/env bash
# R5 / gate G9 -- a work queue of (design, m, host) tasks, kept full.
#   scripts/r5_queue.sh <slots> <workers_each> <<< "B 20 0
#                                                  B 20 1
#                                                  ..."
# or:  scripts/r5_queue.sh <slots> <workers_each> tasks.txt
#
# Why a queue and not run_r5{,_par}.sh: those fix the worker count at launch, so
# the machine drains.  run_r5.sh does one host at a time and idles through each
# host's expensive tail (r = 40, 60, 100); run_r5_par.sh runs N hosts at once but
# shrinks as hosts finish and cannot start the next m until the current one is
# fully done.  Measured: 14 of 64 cores on cell6.  `xargs -P` refills a slot the
# moment a task ends, and the task list spans every remaining m, so the machine
# stays full until the whole list is done.
#
# Blocks are resumable, so listing an already-finished (m, host) is nearly free:
# every block is skipped.  That makes the list safe to over-specify.
set -euo pipefail
SLOTS=$1; WORKERS=$2; SRC=${3:-/dev/stdin}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONPATH=src WORKERS
mkdir -p logs

run_task() {
  local DESIGN=$1 M=$2 H=$3
  local RH CHUNK GRID
  if [ "$DESIGN" = "A" ]; then RH=$(python3 -c "print(int(round(2.4*$M)))"); else RH=24; fi
  # small m: a block of 10 MILPs is milliseconds, so the NFS parquet write
  # dominates -- use fewer, bigger blocks.  Large m: keep blocks small for balance.
  if [ "$M" -le 14 ]; then CHUNK=50; else CHUNK=10; fi
  GRID=$(python3 -c "
back={1,2,3,5,8,10,12,15,20,25,30,40,60,100}
fine=set(range(max(1,$M-1-3), $M-1+6+1))
print(','.join(str(x) for x in sorted(back|fine)))
")
  local ARGS=(--design config/design_r5.yaml --host-id "$H" --r-grid "$GRID"
              --n-s 100 --workers "$WORKERS" --chunk "$CHUNK" --save-vectors all
              --run-id "r5${DESIGN}_m${M}_h${H}")
  # m = 10 with r^H = 24 is the run-1 / A4 setting: land in post_paper{,_h<id>}
  if [ "$M" -ne 10 ] || [ "$RH" -ne 24 ]; then ARGS+=(--m "$M" --r-h "$RH"); fi
  nice -n 19 ionice -c 3 ./.venv/bin/python -m grhd.cli run "${ARGS[@]}" \
    > "logs/task_r5${DESIGN}_m${M}_h${H}.out" 2>&1
  echo "TASK_DONE ${DESIGN} m=${M} h=${H}"
}
export -f run_task

grep -v '^[[:space:]]*$' "$SRC" | xargs -P "$SLOTS" -L 1 bash -c 'run_task "$@"' _
echo "QUEUE_DONE"
