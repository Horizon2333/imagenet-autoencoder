#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# settings
MODEL_ARC=$1
CKPT=$2
DATASET=$3

# CUDA_LAUNCH_BLOCKING=1
python3 -u eval.py \
    --arch $MODEL_ARC \
    --val_list list/${DATASET}_list.txt \
    --workers 16 \
    --batch-size 128 \
    --print-freq 10 \
    --resume ${CKPT}