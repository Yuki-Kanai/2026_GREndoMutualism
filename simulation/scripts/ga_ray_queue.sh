#!/usr/bin/env bash
# Wait for this host's ga_full sweep to finish, then run ga_ray over the same r
# grid.  There is no free capacity to run both at once (2026-09-22), and the ray
# mode costs 1.5-2x a full block, so queueing beats oversubscribing.
#
# Launch detached, with a host-unique log name (the home is shared across cell2/
# cell3/cell6):
#   setsid nohup bash scripts/ga_ray_queue.sh "<r,grid>" <workers> \
#       > logs/ga_ray_queue_sN.log 2>&1 < /dev/null &
set -u
GRID="$1"
WORKERS="$2"
cd "$(dirname "$0")/.."

# The bracket keeps this script's own command line from matching.
while pgrep -f 'ga_curve[.]py .*ga_[f]ull' > /dev/null; do
    sleep 300
done
echo "ga_full finished on $(hostname) at $(date -Is); starting ga_ray"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONPATH=src
exec nice -n 19 ionice -c 3 ./.venv/bin/python scripts/ga_curve.py \
    --methods ga_ray --n-sym 100 --workers "$WORKERS" --r-grid "$GRID"
