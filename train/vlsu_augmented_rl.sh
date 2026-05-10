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