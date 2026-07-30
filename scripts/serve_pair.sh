#!/usr/bin/env bash
# Serve the stock chat model and a merged SDF checkpoint side by side on one H200.
#
# Two servers instead of one so before/after comparisons need no reload, and so the chat app
# can offer an A/B switch. A 9B in bf16 is ~18 GiB of weights, so 0.42 utilisation each
# (~60 GiB) leaves plenty of KV cache at 8k context.
#
#     scripts/serve_pair.sh ckpts/qwen3.5-9b-kirk-1ep kirk-1ep
#
# Stock lands on :8010 as "stock", the checkpoint on :8011 under the name given.
set -euo pipefail

CKPT=${1:?usage: serve_pair.sh <merged-ckpt-dir> <served-name>}
NAME=${2:?usage: serve_pair.sh <merged-ckpt-dir> <served-name>}
CONDA=${CONDA:-$HOME/miniconda3}
LOGDIR=${LOGDIR:-/tmp}
export PATH="$CONDA/envs/vllm-gptoss/bin:$PATH"

# --gdn-prefill-backend triton: the default FlashInfer gated-delta-net path JIT-compiles and
# needs ninja + nvcc, which this box does not have (EngineDeadError on startup otherwise).
common=(--gpu-memory-utilization 0.42 --max-model-len 8192
        --language-model-only --gdn-prefill-backend triton
        --reasoning-parser qwen3 --host 127.0.0.1)

vllm serve Qwen/Qwen3.5-9B --served-model-name stock --port 8010 \
     "${common[@]}" > "$LOGDIR/vllm-stock.log" 2>&1 &
vllm serve "$CKPT" --served-model-name "$NAME" --port 8011 \
     "${common[@]}" > "$LOGDIR/vllm-$NAME.log" 2>&1 &

for port in 8010 8011; do
  echo "waiting for :$port ..."
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null; then echo "  :$port up"; break; fi
    sleep 5
  done
done
curl -s http://127.0.0.1:8010/v1/models | head -c 120; echo
curl -s http://127.0.0.1:8011/v1/models | head -c 120; echo
