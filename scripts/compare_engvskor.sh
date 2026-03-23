#!/bin/bash

MODEL_PATH=${1:-"yueliu1999/GuardReasoner-VL-3B"}
BENCHMARK_PATH=${2:-"./data/benchmark/"}
GPUS=${3:-"4"}

CUDA_VISIBLE_DEVICES=$GPUS python generate.py \
    --model_path $MODEL_PATH \
    --benchmark_path $BENCHMARK_PATH

CUDA_VISIBLE_DEVICES=$GPUS python translate.py \
    --benchmark_path $BENCHMARK_PATH

CUDA_VISIBLE_DEVICES=$GPUS python generate.py \
    --model_path $MODEL_PATH \
    --benchmark_path $BENCHMARK_PATH \
    --suffix _ko

CUDA_VISIBLE_DEVICES=$GPUS python evaluate.py