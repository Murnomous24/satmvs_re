#!/bin/bash

DATASET_ROOT="./data/whu_tlc"
LOG_DIR="./checkpoints"

MODE="train"
MODEL="casmvs"
GEO_MODEL="rpc"
GPU_ID="0"
BATCH_SIZE=1
MIN_INTERVAL=0.5
EPOCH=1000
# TRAIN_LOADCKPT="/home/murph_dl/Paper_Re/SatMVS_Re/checkpoints/casmvs/rpc/20260113_192816/model_000227.ckpt"
# RESUME=True

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
    --epochs=$EPOCH \
    # --train_loadckpt="$TRAIN_LOADCKPT" \
    # --resume=$RESUME