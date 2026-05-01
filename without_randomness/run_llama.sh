#!/bin/bash
# Direct fine-tune of Llama-3.2-1B-Instruct (no randomness factor).
# Edit the values below for your environment, then run: bash run_llama.sh

NPROC=4
TRAIN_DATA="train_data/train.json"
VAL_DATA="train_data/valid.json"
OUT_DIR="out_llama_no_random"

torchrun --standalone --nproc_per_node=$NPROC train.py \
    --backbone llama \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --init pretrained \
    --phase 1 \
    --step 3 \
    --num_iterations 3 \
    --neuron_dim_t 0 \
    --neuron_dim_s 4 \
    --neuron_dim_r 0 \
    --max_iters 200 \
    --batch_size 4 \
    --gradient_accumulation_steps 64 \
    --learning_rate 1e-6 \
    --min_lr 1e-7 \
    --warmup_iters 2 \
    --L1_weight 1e-4 \
    --L1_TARGET 1.0 \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --out_dir $OUT_DIR \
    --task gsm8k-stage
