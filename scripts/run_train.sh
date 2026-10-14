#!/usr/bin/env bash
# Public LoRA recipe; paths and device count are supplied by the caller.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# < 4 )); then
  echo "Usage: $0 BASE_ID BASE_REVISION DATA_JSONL OUTPUT_DIR [kev.train options...]" >&2
  exit 2
fi
base=$1 revision=$2 data=$3 out=$4; shift 4
n=${NPROC_PER_NODE:-1}
if ! [[ $n =~ ^[1-8]$ ]] || (( 8 % n != 0 )); then
  echo 'NPROC_PER_NODE must be 1, 2, 4, or 8 for effective batch 64.' >&2; exit 2
fi
launch=(python -m kev.train)
if (( n > 1 )); then launch=(torchrun --standalone --nproc_per_node "$n" -m kev.train); fi
exec "${launch[@]}" --base "$base" --base_revision "$revision" --data "$data" \
  --suite evals/v7/decision-v7 --replay 12576 --out "$out" \
  --lora 16 --lr 1e-4 --epochs 1 --batch 8 --accum "$((8/n))" \
  --dtype bf16 --checkpointing 1 --device cuda --length_sort 1 \
  --max_state 5120 --p_none_pair 0.25 "$@"
