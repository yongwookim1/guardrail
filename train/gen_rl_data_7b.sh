export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_V1=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=INFO

GUARDREASONER=$(cd "$(dirname "$0")/.." && pwd)
LLAMA_FACTORY=${LLAMA_FACTORY:-$(cd "$GUARDREASONER/.." && pwd)/LLaMA-Factory}
EASYR1=$GUARDREASONER/train/EasyR1

MODEL_PATH=$LLAMA_FACTORY/saves/Custom/full/Qwen2.5-VL-7B/R-SFT-7B
DATA_PATH=$LLAMA_FACTORY/data/

cd $GUARDREASONER/train

CUDA_VISIBLE_DEVICES=0,1,2,3 python rejection_sampling.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --tensor_parallel_size 4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python data_augmentation.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH

mkdir -p $EASYR1/data
mv ./7b_aug_train.parquet $EASYR1/data/
mv ./7b_aug_val.parquet $EASYR1/data/
