#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# settings
MODEL_ARC=$1
DATASET=$2
OUTPUT=results/${DATASET}-${MODEL_ARC}/
mkdir -p ${OUTPUT}

# CUDA_LAUNCH_BLOCKING=1
python3 -u train.py \
    --arch $MODEL_ARC \
    --train_list list/${DATASET}_list.txt \
    --workers 16 \
    --epochs 100 \
    --start-epoch 0 \
    --batch-size 256 \
    --learning-rate 0.05 \
    --momentum 0.9 \
    --weight-decay 1e-4 \
    --print-freq 10 \
    --pth-save-fold ${OUTPUT} \
    --pth-save-epoch 1 \
    --parallel 1 \
    --dist-url 'tcp://localhost:10001' 2>&1 | tee ${OUTPUT}/output.log 