#!/bin/bash
# Step 1: Translate benchmark datasets to Korean using Qwen3.5-9B-Instruct
python translate.py \
    --model_path Qwen/Qwen3.5-9B \
    --benchmark_path ./data/benchmark/ \
    --batch_size 4 \
    --max_new_tokens 4096

# Step 2: Run GuardReasoner-VL inference on Korean-translated inputs
CUDA_VISIBLE_DEVICES=0 python generate.py --model_path yueliu1999/GuardReasoner-VL-7B --suffix _ko
CUDA_VISIBLE_DEVICES=0 python generate.py --model_path yueliu1999/GuardReasoner-VL-Eco-7B --suffix _ko
CUDA_VISIBLE_DEVICES=0 python generate.py --model_path yueliu1999/GuardReasoner-VL-3B --suffix _ko
CUDA_VISIBLE_DEVICES=0 python generate.py --model_path yueliu1999/GuardReasoner-VL-Eco-3B --suffix _ko

# Step 3: Evaluate
python evaluate.py
