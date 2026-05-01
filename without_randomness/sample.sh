#!/bin/bash
# Sample / evaluate a no-randomness checkpoint.
# Usage:
#   BACKBONE=qwen|llama bash sample.sh <checkpoint.pt> <input.jsonl> [extra args]
set -euo pipefail

CKPT="${1:?usage: sample.sh <checkpoint.pt> <input.jsonl> [extra args]}"
INPUT="${2:?usage: sample.sh <checkpoint.pt> <input.jsonl> [extra args]}"
BACKBONE="${BACKBONE:-llama}"
NPROC="${NPROC:-1}"

torchrun --standalone --nproc_per_node="${NPROC}" sample.py \
    --backbone "${BACKBONE}" \
    --checkpoint "${CKPT}" \
    --input_file "${INPUT}" \
    --num_iterations 3 \
    --neuron_dim_t 0 \
    --neuron_dim_s 4 \
    --neuron_dim_r 0 \
    --batch_size 32 \
    --max_new_tokens 100 \
    --temperature 0 \
    "${@:3}"
