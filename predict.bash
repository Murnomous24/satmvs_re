#!/bin/bash

DATASET_ROOT="./data/whu_tlc"
LOG_DIR="./checkpoints"

MODEL="casmvs"
GEO_MODEL="rpc"
GPU_ID="0"
BATCH_SIZE=1
MIN_INTERVAL=0.5
LOADCKPT="/home/murph_dl/Paper_Re/train_log/26_1_31_23_42/model_000005.ckpt"

echo "Starting SatMVS predicting: Model=$MODEL, Geo=$GEO_MODEL, GSD=$MIN_INTERVAL"

python predict.py \
    --model="$MODEL" \
    --geo_model="$GEO_MODEL" \
    --dataset_root="$DATASET_ROOT" \
    --loadckpt="$LOADCKPT" \
    --logdir="$LOG_DIR" \
    --gpu_id="$GPU_ID" \
    --batch_size=$BATCH_SIZE \
    --min_interval=$MIN_INTERVAL \