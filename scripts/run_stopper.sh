#!/usr/bin/env bash
# Offline evaluation: all depth logits have already been computed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rows=${1:?rows directory}; out=${2:?output directory}; shift 2
exec python analysis/adaptive_depth.py --rows "$rows" --out "$out" \
  --calibration decision-v7-calibration --panel eval-v1-development \
  --rule agree --fit xfit --min_agree 0.98 "$@"
