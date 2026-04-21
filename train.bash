#!/bin/bash

DATASET_ROOT="./data/whu_tlc"
LOG_DIR="./checkpoints"

MODE="train"
MODEL="casmvs"
GEO_MODEL="rpc"
GPU_ID="0"
BATCH_SIZE=1
MIN_INTERVAL=0.5
EPOCH=5
PROGRESS_MODE="log"        # options: tqdm | log
PROGRESS_LOG_FREQ=100
NUM_WORKERS=8

# Loss config
LOSS_TYPE="casmvs"         # options: casmvs | entropy | casmvs_entropy | casmvs_dds | entropy_dds
DLOSSW="0.5,1.0,2.0"
ELOSSW="0.5,1.0,2.0"       # used by entropy / casmvs_entropy
ENTROPY_WEIGHT="0.1"       # used by casmvs_entropy
DDSLW="0.5,1.0,2.0"        # used by casmvs_dds
DDS_WEIGHT="0.1"          # used by casmvs_dds
DDS_NUM_BINS="32"          # used by casmvs_dds
DDS_SIGMA="0.0"            # <=0 means auto

# Optional resume config
TRAIN_LOADCKPT=""
RESUME="false"

echo "Starting SatMVS training:"
echo "  Model=$MODEL, Geo=$GEO_MODEL, GSD=$MIN_INTERVAL"
echo "  LossType=$LOSS_TYPE, dlossw=$DLOSSW, elossw=$ELOSSW, entropy_weight=$ENTROPY_WEIGHT, ddslw=$DDSLW, dds_weight=$DDS_WEIGHT"
echo "  ETA=true"

cmd=(
    python train.py
    --mode="$MODE"
    --model="$MODEL"
    --geo_model="$GEO_MODEL"
    --dataset_root="$DATASET_ROOT"
    --logdir="$LOG_DIR"
    --gpu_id="$GPU_ID"
    --batch_size="$BATCH_SIZE"
    --min_interval="$MIN_INTERVAL"
    --epochs="$EPOCH"
    --progress_mode="$PROGRESS_MODE"
    --progress_log_freq="$PROGRESS_LOG_FREQ"
    --num_workers="$NUM_WORKERS"
    --loss_type="$LOSS_TYPE"
    --dlossw="$DLOSSW"
    --eta
)

if [[ "$LOSS_TYPE" == "entropy" || "$LOSS_TYPE" == "casmvs_entropy" || "$LOSS_TYPE" == "entropy_dds" ]]; then
    cmd+=(--elossw="$ELOSSW")
fi

if [[ "$LOSS_TYPE" == "casmvs_entropy" ]]; then
    cmd+=(--entropy_weight="$ENTROPY_WEIGHT")
fi

if [[ "$LOSS_TYPE" == "casmvs_dds" || "$LOSS_TYPE" == "entropy_dds" ]]; then
    cmd+=(
        --ddslw="$DDSLW"
        --dds_weight="$DDS_WEIGHT"
        --dds_num_bins="$DDS_NUM_BINS"
        --dds_sigma="$DDS_SIGMA"
    )
fi

if [[ "$LOSS_TYPE" == "entropy_dds" ]]; then
    cmd+=(--entropy_weight="$ENTROPY_WEIGHT")
fi

if [[ -n "$TRAIN_LOADCKPT" ]]; then
    cmd+=(--train_loadckpt="$TRAIN_LOADCKPT")
fi

if [[ "$RESUME" == "true" ]]; then
    cmd+=(--resume)
fi

"${cmd[@]}"
