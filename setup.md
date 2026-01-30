# vLLM Local Dev Setup (Blackwell GPU - sm_120)

## Step 1: Create Python Environment

```bash
cd /home/ubuntu/workspace/vllm
rm -rf .venv
uv venv --python 3.11
source .venv/bin/activate

# Set CUDA 12.8 paths
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Install torch with CUDA 12.8
export UV_INDEX_STRATEGY="unsafe-best-match"
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu128

# Install xformers (for optimized attention)
uv pip install xformers==0.0.33.post2 --extra-index-url https://download.pytorch.org/whl/cu128

# Fix: xformers downgrades NCCL, but torch cu128 requires NCCL 2.27+
uv pip install nvidia-nccl-cu12==2.27.5 --no-deps --reinstall

# Install build requirements
uv pip install cmake ninja packaging wheel "setuptools>=77.0.3,<80.0.0" setuptools-scm jinja2 regex

# Install runtime dependencies (transformers 5.x breaks vLLM)
uv pip install "transformers>=4.47,<5.0" accelerate sentencepiece tokenizers protobuf openai
```

## Step 2: Build vLLM from Source

```bash
cd /home/ubuntu/workspace/vllm
source .venv/bin/activate

# Set CUDA 12.8 paths
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda-12.8/bin/nvcc
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Build only for Blackwell (sm_120)
export TORCH_CUDA_ARCH_LIST="12.0"

# Enable ccache (critical for rebuilds)
export CCACHE_DIR=/home/ubuntu/.cache/ccache
export CCACHE_NOHASHDIR="true"

# Clean previous builds
rm -rf build .deps vllm/*.so vllm/_C*.so

# Build (~20 min first time, faster with ccache)
# Use --no-deps to prevent NCCL from being downgraded during install
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1 MAX_JOBS=8 uv pip install --no-build-isolation --no-deps -e .
```

---

## Test: Verify Local Code

```bash
cd /home/ubuntu/workspace/vllm
source .venv/bin/activate
python test_qwen3_inference.py
```

Look for this log line to confirm local vLLM code is being used:

```
INFO ... [qwen3.py:53] === QWEN3 MODEL LOADED FROM LOCAL VLLM CODE ===
```

This marker is in `vllm/model_executor/models/qwen3.py` at line 53.
