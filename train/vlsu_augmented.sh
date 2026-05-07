#!/bin/bash
set -e

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export VLLM_USE_V1=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=INFO

GUARDREASONER=$(cd "$(dirname "$0")/.." && pwd)
LLAMA_FACTORY=${LLAMA_FACTORY:-$(cd "$GUARDREASONER/.." && pwd)/LLaMA-Factory}
EASYR1=$GUARDREASONER/train/EasyR1

MODEL_SIZE=${1:-3b}
MODEL_SIZE_LOWER=$(printf "%s" "$MODEL_SIZE" | tr '[:upper:]' '[:lower:]')
MODEL_SIZE_UPPER=$(printf "%s" "$MODEL_SIZE" | tr '[:lower:]' '[:upper:]')

case "$MODEL_SIZE_LOWER" in
    3b)
        MODEL_NAME="Qwen2.5-VL-3B"
        BATCH_SIZE=6
        ;;
    7b)
        MODEL_NAME="Qwen2.5-VL-7B"
        BATCH_SIZE=6
        ;;
    *)
        echo "Usage: bash train/vlsu_augmented.sh [3b|7b]"
        exit 1
        ;;
esac

DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
GPU_COUNT=$(echo "$DEVICES" | awk -F',' '{print NF}')
DATASET_DIR=$LLAMA_FACTORY/data
BASE_MODEL=$GUARDREASONER/models/${MODEL_NAME}-Instruct
SFT_SAVE_PATH=$LLAMA_FACTORY/saves/Custom/full/$MODEL_NAME/R-SFT-${MODEL_SIZE_UPPER}-VLSU
EXPERIMENT=GuardReasoner-VL-${MODEL_SIZE_UPPER}-VLSU
CKPT_DIR=$EASYR1/checkpoints
TRAIN_PARQUET=$EASYR1/data/${MODEL_SIZE_LOWER}_vlsu_aug_train.parquet
VAL_PARQUET=$EASYR1/data/${MODEL_SIZE_LOWER}_vlsu_aug_val.parquet

test -f "$DATASET_DIR/GuardReasoner-VLTrainVLSU.json" || {
    echo "Missing $DATASET_DIR/GuardReasoner-VLTrainVLSU.json"
    exit 1
}

test -d "$DATASET_DIR/vlsu_image" || {
    echo "Missing $DATASET_DIR/vlsu_image"
    exit 1
}

ln -sfn "$DATASET_DIR/vlsu_image" "$LLAMA_FACTORY/vlsu_image"

python3 - "$DATASET_DIR/dataset_info.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text()) if path.exists() else {}
data["GuardReasoner_VLTrainVLSU"] = {
    "file_name": "GuardReasoner-VLTrainVLSU.json",
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system",
    },
}
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
PY

echo "=== Step 1/4: VLSU-augmented R-SFT ==="
cd "$LLAMA_FACTORY"
CUDA_VISIBLE_DEVICES=$DEVICES llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path "$BASE_MODEL" \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template qwen2_vl \
    --flash_attn auto \
    --dataset_dir "$DATASET_DIR" \
    --media_dir "$LLAMA_FACTORY" \
    --dataset GuardReasoner_VLTrainImage,GuardReasoner_VLTrainText,GuardReasoner_VLTrainTextImage,GuardReasoner_VLTrainVLSU \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 200000 \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 1000000 \
    --warmup_steps 0 \
    --packing True \
    --report_to none \
    --output_dir "$SFT_SAVE_PATH" \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --optim adamw_torch_fused \
    --deepspeed "$GUARDREASONER/train/cache/ds_z3_config.json"

echo "=== Step 2/4: Rejection sampling with VLSU ==="
cd "$GUARDREASONER/train"
CUDA_VISIBLE_DEVICES=$DEVICES python rejection_sampling.py \
    --model_path "$SFT_SAVE_PATH" \
    --data_path "$DATASET_DIR/" \
    --tensor_parallel_size "$GPU_COUNT" \
    --include_vlsu

echo "=== Step 3/4: Build RL parquet data with VLSU ==="
CUDA_VISIBLE_DEVICES=$DEVICES python data_augmentation.py \
    --model_path "$SFT_SAVE_PATH" \
    --data_path "$DATASET_DIR/" \
    --include_vlsu \
    --output_prefix "${MODEL_SIZE_LOWER}_vlsu"

mkdir -p "$EASYR1/data"
mv "./${MODEL_SIZE_LOWER}_vlsu_aug_train.parquet" "$TRAIN_PARQUET"
mv "./${MODEL_SIZE_LOWER}_vlsu_aug_val.parquet" "$VAL_PARQUET"

echo "=== Step 4/4: Online RL ==="
SYSTEM_PROMPT="You are a classifier for analyzing interactions between humans and AI.

Given a request from a human user and a response from an AI assistant, finish two tasks.

Task 1, determine whether the human user's request is harmful or unharmful.
Task 2, determine whether the AI assistant's response is harmful or unharmful.

You must think step by step. Keep consistency between the reasoning and the Answers.

Put the reasoning process into <think> </think>. Put the result into <result> </result>."

cd "$EASYR1"
CUDA_VISIBLE_DEVICES=$DEVICES python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.system_prompt="$SYSTEM_PROMPT" \
    data.max_prompt_length=3072 \
    data.max_response_length=3072 \
    worker.actor.model.model_path="$SFT_SAVE_PATH" \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=32 \
    worker.reward.reward_type="function" \
    worker.reward.compute_score="safety" \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=16 \
    worker.rollout.temperature=1.2 \
    worker.rollout.gpu_memory_utilization=0.5 \
    trainer.experiment_name="$EXPERIMENT" \
    trainer.total_episodes=1 \
    trainer.n_gpus_per_node="$GPU_COUNT" \
    trainer.save_checkpoint_path="$CKPT_DIR/$EXPERIMENT"

LATEST_CKPT=$(find "$CKPT_DIR/$EXPERIMENT" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -1)
python3 scripts/model_merger.py --local_dir "$LATEST_CKPT/actor"

echo "VLSU-augmented model saved under:"
echo "$LATEST_CKPT/actor/huggingface"
