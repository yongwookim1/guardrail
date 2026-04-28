#!/bin/bash
set -e

WORK_DIR=${WORK_DIR:-$HOME/MLLM_Safety}
GUARDREASONER=$WORK_DIR/guardreasoner
LLAMA_FACTORY=$WORK_DIR/LLaMA-Factory
EASYR1=$GUARDREASONER/train/EasyR1
DATA_DIR=$GUARDREASONER/data/train/llamafactory_data

echo "=== Step 1: Check PyTorch ==="
if python -c "import torch" 2>/dev/null; then
    python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
else
    echo "PyTorch not found. Install it manually before running this script:"
    echo "  conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y"
    exit 1
fi

echo "=== Step 2: pip dependencies ==="
pip install \
    "transformers>=4.54.0,<5.0.0" \
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
    numpy \
    pandas \
    scikit-learn \
    psutil

echo "=== Step 3: Clone and install LLaMA-Factory ==="
if [ -d "$LLAMA_FACTORY" ]; then
    echo "LLaMA-Factory already exists at $LLAMA_FACTORY — skipping clone"
else
    git clone https://github.com/hiyouga/LLaMA-Factory $LLAMA_FACTORY
fi
cd $LLAMA_FACTORY
pip install -e ".[torch,metrics]"
cd -

echo "=== Step 4: Wire existing data into LLaMA-Factory ==="
rm -rf $LLAMA_FACTORY/data
ln -s $DATA_DIR $LLAMA_FACTORY/data

echo "=== Step 5: Create DeepSpeed ZeRO-3 config ==="
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

echo "=== Step 6: PYTHONPATH for EasyR1/verl ==="
grep -qF "EasyR1" ~/.bashrc || \
    echo "export PYTHONPATH=$EASYR1:\$PYTHONPATH" >> ~/.bashrc

echo "=== Step 7: Install flash-attn via pre-built wheel (bypasses GCC 13 compile issue) ==="
PY_VER=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
FLASH_WHEEL="flash_attn-2.7.4+cu124torch2.5cxx11abiFALSE-${PY_VER}-${PY_VER}-linux_x86_64.whl"
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4/${FLASH_WHEEL}"

echo "Detected Python tag: $PY_VER — downloading $FLASH_WHEEL"
pip install "$FLASH_URL" || {
    echo "Pre-built wheel not found for $PY_VER. Falling back to source build with gcc-12..."
    sudo apt-get install -y gcc-12 g++-12
    CC=gcc-12 CXX=g++-12 pip install flash-attn --no-build-isolation
}

echo ""
echo "Done. Run: source ~/.bashrc"
echo "Verify: python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'"
