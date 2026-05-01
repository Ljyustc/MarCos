#!/bin/bash
# Phase-2 training, Qwen2.5-0.5B backbone (with randomness factor).
# Trains the random predictor on top of a frozen Phase-1 checkpoint.
# Edit RESUME_CKPT to point at your Phase-1 checkpoint, then run: bash run_qwen_phase2.sh

NPROC=4
TRAIN_DATA="train_data/train.json"
VAL_DATA="train_data/valid.json"
OUT_DIR="out_qwen"
RESUME_CKPT="out_qwen/phase1_saved.pt"

torchrun --standalone --nproc_per_node=$NPROC train.py \
    --backbone qwen \
    --model_path Qwen/Qwen2.5-0.5B \
    --phase 2 \
    --init resume \
    --resume_ckpt $RESUME_CKPT \
    --step 3 \
    --num_iterations 3 \
    --neuron_dim_t 10 \
    --neuron_dim_s 100 \
    --neuron_dim_r 16 \
    --max_iters 200 \
    --batch_size 64 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --warmup_iters 2 \
    --L1_weight 1e-4 \
    --L1_TARGET 10.0 \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --out_dir $OUT_DIR \
    --task gsm-stage
