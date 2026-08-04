#!/usr/bin/env bash
# Serve BOTH arms from one 67 GB base model: `stock` is the base weights, `armP` applies a LoRA
# adapter. Two merged 35B checkpoints are 134 GB and will not fit on one H200 with any KV cache.
#
# The adapter must be key-remapped by scripts/adapter_for_vllm.py first, or vllm silently applies
# nothing -- see chat/README.md. This script checks for the remapped directory and refuses
# otherwise, because the failure mode is two identical models rather than an error.
set -euo pipefail

ADAPTER=${1:-runs/news2026-armP/adapter-vllm}
NAME=${2:-armP}
BASE=${3:-Qwen/Qwen3.5-35B-A3B}
CONDA=${CONDA:-$HOME/miniconda3}
export PATH="$CONDA/envs/vllm-gptoss/bin:$PATH"

[ -f "$ADAPTER/adapter_model.safetensors" ] || {
  echo "no adapter at $ADAPTER -- run scripts/adapter_for_vllm.py first" >&2; exit 1; }
grep -q "language_model" <(python - <<'EOF'
from safetensors import safe_open
import sys, os
with safe_open(os.environ["ADAPTER"] + "/adapter_model.safetensors", "pt") as f:
    print(next(iter(f.keys())))
EOF
) || { echo "adapter keys are not remapped for serving -- run scripts/adapter_for_vllm.py" >&2; exit 1; }

vllm serve "$BASE" --served-model-name stock --port 8010 \
  --enable-lora --max-lora-rank 32 --lora-modules "$NAME=$ADAPTER" \
  --gpu-memory-utilization 0.80 --max-model-len 32768 --language-model-only \
  --gdn-prefill-backend triton --reasoning-parser qwen3 --host 127.0.0.1
