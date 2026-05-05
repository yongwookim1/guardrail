#!/bin/bash
set -e

GUARDREASONER=$(cd "$(dirname "$0")" && pwd)
LLAMA_FACTORY=${LLAMA_FACTORY:-$(cd "$GUARDREASONER/.." && pwd)/LLaMA-Factory}
EASYR1=$GUARDREASONER/train/EasyR1
DATA_DIR=$GUARDREASONER/data/train/llamafactory_data

echo "=== Verify PyTorch ==="
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPUs:', torch.cuda.device_count())"

echo "=== Install pip dependencies ==="
pip install \
    "transformers>=4.56.0" \
    "tokenizers>=0.21.1" \
    "numpy>=2" \
    accelerate \
    datasets \
    peft \
    "trl>=0.18.0" \
    deepspeed \
    liger-kernel \
    mathruler \
    omegaconf \
    tensordict \
    torchdata \
    qwen-vl-utils \
    "ray[default]" \
    "vllm>=0.8.0" \
    wandb \
    "pyarrow>=15.0.0" \
    pillow \
    "av>=10.0.0" \
    einops \
    scipy \
    sentencepiece \
    tiktoken \
    safetensors \
    fire \
    pylatexenc \
    codetiming \
    pydantic \
    anthropic \
    openai \
    tqdm \
    packaging \
    protobuf \
    pyyaml \
    pandas \
    scikit-learn \
    psutil

echo "=== Install LLaMA-Factory (latest) ==="
if [ ! -d "$LLAMA_FACTORY" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory $LLAMA_FACTORY
fi
cd $LLAMA_FACTORY
pip install -e ".[torch,metrics]"
cd -

echo "=== Wire data into LLaMA-Factory ==="
rm -rf $LLAMA_FACTORY/data
ln -s $DATA_DIR $LLAMA_FACTORY/data

echo "=== Fix image paths ==="
ln -sfn $DATA_DIR/image $GUARDREASONER/data/image
ln -sfn $DATA_DIR/text_image $GUARDREASONER/data/text_image

echo "=== Write dataset_info.json ==="
python3 -c "
import json
d = {
  'GuardReasoner_VLTrainImage': {'file_name': 'GuardReasoner-VLTrainImage.json', 'formatting': 'sharegpt', 'columns': {'messages': 'messages', 'images': 'images'}, 'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant', 'system_tag': 'system'}},
  'GuardReasoner_VLTrainText': {'file_name': 'GuardReasoner-VLTrainText.json', 'formatting': 'sharegpt', 'columns': {'messages': 'messages'}, 'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant', 'system_tag': 'system'}},
  'GuardReasoner_VLTrainTextImage': {'file_name': 'GuardReasoner-VLTrainTextImage.json', 'formatting': 'sharegpt', 'columns': {'messages': 'messages', 'images': 'images'}, 'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant', 'system_tag': 'system'}}
}
json.dump(d, open('$DATA_DIR/dataset_info.json', 'w'), indent=2)
"

echo "=== Create DeepSpeed ZeRO-3 config ==="
mkdir -p $GUARDREASONER/train/cache
cat > $GUARDREASONER/train/cache/ds_z3_config.json << 'EOF'
{
  "bf16": { "enabled": "auto" },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 100,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
EOF

echo "=== Set PYTHONPATH ==="
grep -qF "EasyR1" ~/.bashrc || \
    echo "export PYTHONPATH=$EASYR1:\$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc

echo "=== Install flash-attn ==="
sudo apt-get install -y gcc-12 g++-12
CC=gcc-12 CXX=g++-12 pip install flash-attn --no-build-isolation || \
    echo "WARNING: flash-attn failed — training still works, just slower"

echo ""
echo "=== Setup complete ==="
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPUs:', torch.cuda.device_count())"
