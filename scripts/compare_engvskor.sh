#!/bin/bash

MODEL_PATH=${1:-"yueliu1999/GuardReasoner-VL-3B"}
BENCHMARK_PATH=${2:-"./data/benchmark/"}
GPUS=${3:-"4"}
export HF_HOME=${HF_HOME:-"/home/hg_models"}
export HF_HUB_OFFLINE=1
MIDM_PATH=${4:-"K-intelligence/Midm-2.0-Base-Instruct"}
if [ -d "$(dirname $MODEL_PATH)" ] && [ "$(dirname $MODEL_PATH)" != "." ]; then
    MIDM_PATH=$(dirname $MODEL_PATH)/Midm-2.0-Base-Instruct
fi

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