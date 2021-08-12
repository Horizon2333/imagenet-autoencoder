#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# settings
MODEL_ARC=$1
FOLDER=$2
DATASET=$3

# CUDA_LAUNCH_BLOCKING=1
python3 -u eval.py \
    --arch $MODEL_ARC \
    --val_list list/${DATASET}_list.txt \
    --workers 16 \
    --batch-size 256 \
    --print-freq 10 \
    --folder ${FOLDER} \
    --start_epoch 0 \
    --epochs 100
