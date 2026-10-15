#!/usr/bin/env bash
# Regenerate every notebook, figure and table from the cache.
#
#   bash run_all.sh            # all notebooks
#   bash run_all.sh 02 04      # only the notebooks whose name starts with 02 / 04
#
# Assumes the GRHDanalysis environment is on PATH (or set ENV_PREFIX below) and
# that the cache exists:  Rscript R/10_build_cache.R
set -euo pipefail

ENV_PREFIX="${GRHD_ENV_PREFIX:-$HOME/miniconda3/envs/GRHDanalysis}"
export PATH="$ENV_PREFIX/bin:$PATH"

# One BLAS thread: this runs on a shared machine and the work is I/O bound.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$(dirname "$0")"
mkdir -p logs

if [ "$#" -gt 0 ]; then
  PATTERNS=("$@")
else
  PATTERNS=("00" "01" "02" "03" "04" "05" "06" "07")
  # A full run regenerates everything, so clear the output directories first.
  # Otherwise a figure that has been renamed or dropped stays on disk looking
  # current, and can end up in the manuscript long after the code stopped
  # producing it.  (A partial run leaves them alone -- it cannot know what is
  # still wanted.)
  echo "full run: clearing figures/ and tables/"
  rm -f figures/*.pdf figures/*.png figures/*.meta.json tables/*.csv
fi

for p in "${PATTERNS[@]}"; do
  for src in notebooks/"$p"*.R; do
    [ -e "$src" ] || continue
    nb="${src%.R}.ipynb"
    name="$(basename "${src%.R}")"
    echo "=== $name"
    python tools/percent2ipynb.py "$src" > /dev/null
    nice -n 19 jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=7200 \
      --ExecutePreprocessor.kernel_name=ir_grhd \
      "$nb" > "logs/$name.log" 2>&1 \
      || { echo "FAILED: see logs/$name.log"; tail -25 "logs/$name.log"; exit 1; }
    echo "    ok -> $nb"
  done
done

echo
echo "figures: $(ls figures/*.pdf 2>/dev/null | wc -l) pdf"
echo "tables : $(ls tables/*.csv 2>/dev/null | wc -l) csv"
