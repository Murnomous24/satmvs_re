#!/bin/bash

DATASET_ROOT="./data/whu_tlc"
LOG_DIR="./checkpoints"

MODE="train"
MODEL="casmvs"
GEO_MODEL="rpc"
GPU_ID="0"
BATCH_SIZE=1
MIN_INTERVAL=0.5

echo "Starting SatMVS training: Model=$MODEL, Geo=$GEO_MODEL, GSD=$MIN_INTERVAL"

python train.py \
    --mode="$MODE" \
    --model="$MODEL" \
    --geo_model="$GEO_MODEL" \
    --dataset_root="$DATASET_ROOT" \
    --logdir="$LOG_DIR" \
    --gpu_id="$GPU_ID" \
    --batch_size=$BATCH_SIZE \
    --min_interval=$MIN_INTERVAL \