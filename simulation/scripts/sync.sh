#!/usr/bin/env bash
# Push the source tree to the cluster (shared /mnt2 home, so cell3 is enough).
set -euo pipefail
REMOTE=${REMOTE:-furusawalab-cell3}
RDIR=${RDIR:-'~/20260915_GRHD_Claude/20260917_GRHD_YKrepro'}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
tar czf - src config tests scripts handover Makefile pyproject.toml 2>/dev/null | \
  ssh "$REMOTE" "mkdir -p $RDIR && tar xzf - -C $RDIR"
echo "synced to $REMOTE:$RDIR"
