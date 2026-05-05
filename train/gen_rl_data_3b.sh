export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GUARDREASONER=$(cd "$(dirname "$0")/.." && pwd)
LLAMA_FACTORY=${LLAMA_FACTORY:-$(cd "$GUARDREASONER/.." && pwd)/LLaMA-Factory}
EASYR1=$GUARDREASONER/train/EasyR1

MODEL_PATH=$LLAMA_FACTORY/saves/Custom/full/Qwen2.5-VL-3B/R-SFT-3B
DATA_PATH=$LLAMA_FACTORY/data/

cd $GUARDREASONER/train

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python rejection_sampling.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --tensor_parallel_size 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python data_augmentation.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH

mkdir -p $EASYR1/data
mv ./3b_aug_train.parquet $EASYR1/data/
mv ./3b_aug_val.parquet $EASYR1/data/
