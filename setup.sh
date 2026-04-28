#!/bin/bash
set -e

WORK_DIR=$HOME/MLLM_Safety
GUARDREASONER=$WORK_DIR/guardreasoner
LLAMA_FACTORY=$WORK_DIR/LLaMA-Factory
EASYR1=$GUARDREASONER/train/EasyR1
DATA_DIR=$GUARDREASONER/data/train/llamafactory_data

echo "=== Remove and recreate gr-train from guardrail-eval ==="
conda deactivate 2>/dev/null || true
conda remove -n gr-train --all -y 2>/dev/null || true
conda create -n gr-train --clone guardrail-eval -y

echo "=== Activate gr-train ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gr-train

echo "=== Verify PyTorch ==="
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| Available:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

echo "=== Install pip dependencies ==="
pip install \
    "transformers>=4.56.0" \
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
    "numpy>=2" \
    pandas \
    scikit-learn \
    psutil \
    "tokenizers>=0.21.1"

echo "=== Clone and install LLaMA-Factory v0.9.1 ==="
if [ -d "$LLAMA_FACTORY" ]; then
    echo "LLaMA-Factory already exists — skipping clone"
else
    git clone --branch v0.9.1 --depth 1 https://github.com/hiyouga/LLaMA-Factory $LLAMA_FACTORY
fi
cd $LLAMA_FACTORY
pip install -e ".[torch,metrics]" --ignore-requires-python
pip install "transformers>=4.56.0" "tokenizers>=0.21.1" --upgrade
cd -

echo "=== Wire data into LLaMA-Factory ==="
rm -rf $LLAMA_FACTORY/data
ln -s $DATA_DIR $LLAMA_FACTORY/data

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

echo "=== Set PYTHONPATH for EasyR1 ==="
grep -qF "EasyR1" ~/.bashrc || \
    echo "export PYTHONPATH=$EASYR1:\$PYTHONPATH" >> ~/.bashrc

echo "=== Install flash-attn (source build with gcc-12, CUDA 12.8 matched) ==="
sudo apt-get install -y gcc-12 g++-12
CC=gcc-12 CXX=g++-12 pip install flash-attn --no-build-isolation || \
    echo "WARNING: flash-attn failed — training will use standard attention (slower but works)"

echo ""
echo "=== Done ==="
source ~/.bashrc
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPUs:', torch.cuda.device_count())"
