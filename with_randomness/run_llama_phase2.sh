#!/bin/bash
# Phase-2 training for the Llama-3.2-1B-Instruct backbone (with randomness factor).
set -euo pipefail

NPROC="${NPROC:-4}"
TRAIN_DATA="${TRAIN_DATA:-train_data/train.json}"
VAL_DATA="${VAL_DATA:-train_data/valid.json}"
OUT_DIR="${OUT_DIR:-out_llama}"
RESUME_CKPT="${RESUME_CKPT:-${1:-out_llama/phase1_saved.pt}}"

torchrun --standalone --nproc_per_node="${NPROC}" train.py \
    --backbone llama \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --phase 2 \
    --init resume \
    --resume_ckpt "${RESUME_CKPT}" \
    --step 3 \
    --num_iterations 3 \
    --neuron_dim_t 10 \
    --neuron_dim_s 100 \
    --neuron_dim_r 16 \
    --max_iters 200 \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --min_lr 1e-6 \
    --warmup_iters 2 \
    --L1_weight 1e-4 \
    --L1_TARGET 10.0 \
    --train_data "${TRAIN_DATA}" \
    --val_data "${VAL_DATA}" \
    --out_dir "${OUT_DIR}" \
    --task gsm-stage \
    "${@:2}"
