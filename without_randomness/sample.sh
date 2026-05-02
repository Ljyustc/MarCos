#!/bin/bash
# Sample / evaluate a no-randomness checkpoint.
# Edit the values below, then run: bash sample.sh

NPROC=1
BACKBONE="llama"
CKPT="out_llama_no_random/your_checkpoint.pt"
INPUT="test.jsonl"

torchrun --standalone --nproc_per_node=$NPROC sample.py \
    --backbone $BACKBONE \
    --checkpoint $CKPT \
    --input_file $INPUT \
    --num_iterations 3 \
    --neuron_dim_t 0 \
    --neuron_dim_s 4 \
    --neuron_dim_r 0 \
    --batch_size 32 \
    --max_new_tokens 100 \
    --temperature 0 \
    --use_chat_template
    # Remove `--use_chat_template` above when sampling from a base-model
    # checkpoint (e.g. trained on Qwen2.5-0.5B without chat template).
