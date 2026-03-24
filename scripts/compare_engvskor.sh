#!/bin/bash

MODEL_PATH=${1:-"yueliu1999/GuardReasoner-VL-3B"}
BENCHMARK_PATH=${2:-"./data/benchmark/"}
GPUS=${3:-"4"}
export HF_HOME=${HF_HOME:-"/home/hg_models"}
export HF_HUB_OFFLINE=1

find_model() { find /home /data /scratch 2>/dev/null -type d -name "$1" | head -1; }

MODEL_PATH=${1:-$(find_model "GuardReasoner-VL*")}
MIDM_PATH=${4:-$(find_model "Midm*")}
MIDM_PATH=${MIDM_PATH:-$(dirname $MODEL_PATH)/Midm-2.0-Base-Instruct}
MIDM_PATH=${MIDM_PATH:-"K-intelligence/Midm-2.0-Base-Instruct"}

[ -z "$MODEL_PATH" ] && { echo "ERROR: Could not find GuardReasoner model."; exit 1; }

CUDA_VISIBLE_DEVICES=$GPUS python generate.py \
    --model_path $MODEL_PATH \
    --benchmark_path $BENCHMARK_PATH

CUDA_VISIBLE_DEVICES=$GPUS python translate.py \
    --model_path $MIDM_PATH \
    --benchmark_path $BENCHMARK_PATH

CUDA_VISIBLE_DEVICES=$GPUS python generate.py \
    --model_path $MODEL_PATH \
    --benchmark_path $BENCHMARK_PATH \
    --suffix _ko

CUDA_VISIBLE_DEVICES=$GPUS python evaluate.py