#!/bin/bash

# Configuration
CKPT_PATH="/home/murph_dl/Paper_Re/train_log/26_1_31_23_42/model_000005.ckpt"
INFO_ROOT="dsm_infos/whu_tlc"
OUTPUT_DIR="./dsm_results"
GPU_ID="0"

# Run prediction
python predict_dsm.py \
    --config_file "dsm_config/config.json" \
    --info_root "$INFO_ROOT" \
    --model casmvs \
    --geo_model rpc \
    --loadckpt "$CKPT_PATH" \
    --resize_scale 1 \
    --sample_scale 1 \
    --interval_scale 1 \
    --batch_size 1 \
    --ndepths "64,32,8" \
    --depth_inter_ratio "4,2,1" \
    --cr_base_chs "8,8,8" \
    --gpu_id "$GPU_ID" \
    --workspace "$OUTPUT_DIR"
