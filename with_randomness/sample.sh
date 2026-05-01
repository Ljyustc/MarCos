#!/bin/bash
# Sample / evaluate from a phase-2 checkpoint.
# Edit the values below, then run: bash sample.sh

NPROC=1
BACKBONE="qwen"
CKPT="out_qwen/phase2_saved.pt"
INPUT="test.jsonl"

torchrun --standalone --nproc_per_node=$NPROC sample.py \
    --backbone $BACKBONE \
    --checkpoint $CKPT \
    --input_file $INPUT \
    --num_iterations 3 \
    --neuron_dim_t 1 \
    --neuron_dim_s 100 \
    --neuron_dim_r 16 \
    --batch_size 64 \
    --max_new_tokens 100 \
    --temperature 0
