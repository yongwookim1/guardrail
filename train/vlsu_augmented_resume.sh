#!/bin/bash
# Resume vlsu_augmented.sh from step 4 (Online RL).
# Assumes steps 1-3 already finished:
#   - SFT checkpoint at $SFT_SAVE_PATH
#   - Parquet data at $TRAIN_PARQUET / $VAL_PARQUET
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
        ;;
    7b)
        MODEL_NAME="Qwen2.5-VL-7B"
        ;;
    *)
        echo "Usage: bash train/vlsu_augmented_resume.sh [3b|7b]"
        exit 1
        ;;
esac

DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
GPU_COUNT=$(echo "$DEVICES" | awk -F',' '{print NF}')
DATASET_DIR=$LLAMA_FACTORY/data
SFT_SAVE_PATH=$LLAMA_FACTORY/saves/Custom/full/$MODEL_NAME/R-SFT-${MODEL_SIZE_UPPER}-VLSU
EXPERIMENT=GuardReasoner-VL-${MODEL_SIZE_UPPER}-VLSU
CKPT_DIR=$EASYR1/checkpoints
TRAIN_PARQUET=$EASYR1/data/${MODEL_SIZE_LOWER}_vlsu_aug_train.parquet
VAL_PARQUET=$EASYR1/data/${MODEL_SIZE_LOWER}_vlsu_aug_val.parquet

# Sanity: artifacts from steps 1-3 must exist.
test -d "$SFT_SAVE_PATH" || { echo "Missing SFT checkpoint: $SFT_SAVE_PATH (run steps 1-3 first)"; exit 1; }
test -f "$TRAIN_PARQUET"  || { echo "Missing parquet: $TRAIN_PARQUET (run step 3 first)"; exit 1; }
test -f "$VAL_PARQUET"    || { echo "Missing parquet: $VAL_PARQUET (run step 3 first)"; exit 1; }

ln -sfn "$DATASET_DIR/vlsu_image" "$LLAMA_FACTORY/vlsu_image"

echo "=== Step 4/7: Online RL ==="
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
    trainer.logger=[console] \
    trainer.n_gpus_per_node="$GPU_COUNT" \
    trainer.save_checkpoint_path="$CKPT_DIR/$EXPERIMENT"

LATEST_CKPT=$(find "$CKPT_DIR/$EXPERIMENT" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -1)
python3 "$EASYR1/scripts/model_merger.py" --local_dir "$LATEST_CKPT/actor"

RL_MODEL_PATH="$LATEST_CKPT/actor/huggingface"
RL_MODEL_LINK="$GUARDREASONER/models/GuardReasoner-VL-${MODEL_SIZE_UPPER}-VLSU"
ln -sfn "$RL_MODEL_PATH" "$RL_MODEL_LINK"

echo "VLSU-augmented model saved under:"
echo "$RL_MODEL_PATH"
echo "Linked as: $RL_MODEL_LINK"

PRETRAINED_MODEL="$GUARDREASONER/models/GuardReasoner-VL-${MODEL_SIZE_UPPER}"
PRETRAINED_NAME="GuardReasoner-VL-${MODEL_SIZE_UPPER}"

if [ ! -d "$PRETRAINED_MODEL" ]; then
    echo "ERROR: Pretrained baseline not found at $PRETRAINED_MODEL"
    echo "Download it (e.g.) with:"
    echo "  huggingface-cli download yueliu1999/$PRETRAINED_NAME --local-dir $PRETRAINED_MODEL"
    exit 1
fi

VLSU_TEST_JSON="$DATASET_DIR/GuardReasoner-VLTestVLSU.json"
test -f "$VLSU_TEST_JSON" || {
    echo "Missing $VLSU_TEST_JSON"
    exit 1
}

echo "=== Step 5/7: VLSU evaluation (Pretrained + SFT + RL) ==="
cd "$GUARDREASONER/train"

VLSU_CSV="$GUARDREASONER/result_vlsu.csv"
rm -f "$VLSU_CSV"

echo "--- VLSU eval: Pretrained ($PRETRAINED_MODEL) ---"
CUDA_VISIBLE_DEVICES=$DEVICES python evaluate_vlsu.py \
    --model_path "$PRETRAINED_MODEL" \
    --dataset "$VLSU_TEST_JSON" \
    --tensor_parallel_size "$GPU_COUNT" \
    --csv_path "$VLSU_CSV"

echo "--- VLSU eval: R-SFT model ---"
CUDA_VISIBLE_DEVICES=$DEVICES python evaluate_vlsu.py \
    --model_path "$SFT_SAVE_PATH" \
    --dataset "$VLSU_TEST_JSON" \
    --tensor_parallel_size "$GPU_COUNT" \
    --csv_path "$VLSU_CSV"

echo "--- VLSU eval: RL-merged model ---"
CUDA_VISIBLE_DEVICES=$DEVICES python evaluate_vlsu.py \
    --model_path "$RL_MODEL_LINK" \
    --dataset "$VLSU_TEST_JSON" \
    --tensor_parallel_size "$GPU_COUNT" \
    --csv_path "$VLSU_CSV"

echo "=== Step 6/7: Full benchmark generation (Pretrained + SFT + RL) ==="
cd "$GUARDREASONER"

echo "--- generate.py: Pretrained ($PRETRAINED_MODEL) ---"
CUDA_VISIBLE_DEVICES=$DEVICES python generate.py --model_path "$PRETRAINED_MODEL"

echo "--- generate.py: R-SFT model ---"
CUDA_VISIBLE_DEVICES=$DEVICES python generate.py --model_path "$SFT_SAVE_PATH"

echo "--- generate.py: RL-merged model ---"
CUDA_VISIBLE_DEVICES=$DEVICES python generate.py --model_path "$RL_MODEL_LINK"

echo "=== Step 7/7: Compute F1 scores (evaluate.py) ==="
cd "$GUARDREASONER"
python evaluate.py

echo ""
echo "Comparison ready. Predictions saved under:"
echo "  ./data/test/$PRETRAINED_NAME/                  (pretrained baseline)"
echo "  ./data/test/$(basename "$SFT_SAVE_PATH")/      (R-SFT)"
echo "  ./data/test/$(basename "$RL_MODEL_LINK")/      (RL)"
echo ""
echo "F1 scores written to:"
echo "  $GUARDREASONER/result.csv      (full benchmarks)"
echo "  $VLSU_CSV  (VLSU)"
