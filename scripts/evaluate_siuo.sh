#!/bin/bash
set -e

GUARDREASONER=$(cd "$(dirname "$0")/.." && pwd)
cd "$GUARDREASONER"

GPUS=${1:-"0,1,2,3"}
SIUO_DIR=${2:-"./data/SIUO/"}
export HF_HOME=${HF_HOME:-"/home/hg_models"}
export HF_HUB_OFFLINE=1

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_V1=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

find_model() { find /home /data /scratch 2>/dev/null -type d -name "$1" | head -1; }

PRETRAINED_PATH=$(find_model "GuardReasoner-VL-3B")
PRETRAINED_PATH=${PRETRAINED_PATH:-"yueliu1999/GuardReasoner-VL-3B"}

SFT_PATH=$(find_model "R-SFT-3B-VLSU")

[ ! -d "$SIUO_DIR" ] && { echo "ERROR: SIUO dataset not found at $SIUO_DIR"; exit 1; }

echo "=== SIUO Evaluation ==="
echo "SIUO dir : $SIUO_DIR"
echo "GPUs     : $GPUS"
echo ""

echo "[1/3] Pretrained: $PRETRAINED_PATH"
CUDA_VISIBLE_DEVICES=$GPUS python generate_siuo.py \
    --model_path "$PRETRAINED_PATH" \
    --siuo_dir "$SIUO_DIR"

if [ -n "$SFT_PATH" ]; then
    echo "[2/3] SFT: $SFT_PATH"
    CUDA_VISIBLE_DEVICES=$GPUS python generate_siuo.py \
        --model_path "$SFT_PATH" \
        --siuo_dir "$SIUO_DIR"
else
    echo "[2/3] SFT model not found, skipping."
fi

echo "[3/3] Evaluating..."
python evaluate_siuo.py
