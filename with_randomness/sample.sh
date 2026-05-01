#!/bin/bash
# Sample / evaluate from a phase-2 checkpoint.
# Usage:
#   bash sample.sh <checkpoint.pt> <test.jsonl> [--backbone qwen|llama]
set -euo pipefail

CKPT="${1:?usage: sample.sh <checkpoint.pt> <input.jsonl> [extra args]}"
INPUT="${2:?usage: sample.sh <checkpoint.pt> <input.jsonl> [extra args]}"
BACKBONE="${BACKBONE:-qwen}"
NPROC="${NPROC:-1}"

torchrun --standalone --nproc_per_node="${NPROC}" sample.py \
    --backbone "${BACKBONE}" \
    --checkpoint "${CKPT}" \
    --input_file "${INPUT}" \
    --num_iterations 3 \
    --neuron_dim_t 1 \
    --neuron_dim_s 100 \
    --neuron_dim_r 16 \
    --batch_size 64 \
    --max_new_tokens 100 \
    --temperature 0 \
    "${@:3}"
