#!/bin/bash
set -e

GUARDREASONER=$(cd "$(dirname "$0")/.." && pwd)
cd "$GUARDREASONER"

MODEL_SIZE=${1:-3b}
GPUS=${2:-"0,1,2,3"}
SIUO_DIR=${3:-"./data/SIUO/"}

MODEL_SIZE_UPPER=$(printf "%s" "$MODEL_SIZE" | tr '[:lower:]' '[:upper:]')
MODEL_NAME="Qwen2.5-VL-${MODEL_SIZE_UPPER}"
LLAMA_FACTORY=${LLAMA_FACTORY:-$(cd "$GUARDREASONER/.." && pwd)/LLaMA-Factory}

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_V1=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

PRETRAINED_PATH="$GUARDREASONER/models/GuardReasoner-VL-${MODEL_SIZE_UPPER}"
SFT_PATH="$LLAMA_FACTORY/saves/Custom/full/$MODEL_NAME/R-SFT-${MODEL_SIZE_UPPER}-VLSU"

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
