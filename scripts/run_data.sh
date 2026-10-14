#!/usr/bin/env bash
# Explicit action required; see docs/DATA.md before invoking builders.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
action=${1:-help}; if (( $# )); then shift; fi
case "$action" in
  prepare-s31) exec python scripts/prepare_training_mix.py "$@" ;;
  eval) exec python scripts/build_lev_eval.py "$@" ;;
  train) exec python scripts/build_lev_train.py "$@" ;;
  *) echo "Usage: $0 {prepare-s31|eval|train} [builder options]"; echo 'Requires licensed local inputs; see docs/DATA.md.'; exit 2 ;;
esac
