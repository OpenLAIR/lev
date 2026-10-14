#!/usr/bin/env bash
# Defaults to development plus separate temperature-calibration records.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run=${1:?checkpoint directory}; out=${2:?output directory}; shift 2
exec python scripts/depth_rows.py --run "$run" --out "$out" \
  --read evals/v7/decision-v7:calibration --read evals/levbench-v1:development "$@"
