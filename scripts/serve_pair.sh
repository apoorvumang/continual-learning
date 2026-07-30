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
# --max-model-len 32768: prompt and output share this budget, so a chat app asking for 8192
# output tokens in thinking mode 400s against a small context. Do not trim it to fit another
# instance on the GPU -- lower --gpu-memory-utilization instead.
common=(--gpu-memory-utilization 0.35 --max-model-len 32768
        --language-model-only --gdn-prefill-backend triton
        --reasoning-parser qwen3 --host 127.0.0.1)

# Started one at a time, not concurrently: vllm requires *free* memory >= utilization x total
# at startup, so two simultaneous launches each see the other's weights and one dies with
# "No available memory for the cache blocks".
serve_one() {  # port, model, name
  vllm serve "$2" --served-model-name "$3" --port "$1" \
       "${common[@]}" > "$LOGDIR/vllm-$3.log" 2>&1 &
  echo "waiting for :$1 ..."
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$1/v1/models" > /dev/null; then echo "  :$1 up"; return; fi
    sleep 5
  done
  echo "  :$1 FAILED -- see $LOGDIR/vllm-$3.log" >&2
  return 1
}

serve_one 8010 Qwen/Qwen3.5-9B stock
serve_one 8011 "$CKPT" "$NAME"
curl -s http://127.0.0.1:8010/v1/models | head -c 120; echo
curl -s http://127.0.0.1:8011/v1/models | head -c 120; echo
