#!/usr/bin/env bash
# Serve the stock chat model and a merged SDF checkpoint side by side on one H200.
#
# Two servers instead of one so before/after comparisons need no reload, and so the chat app
# can offer an A/B switch. Utilisation is set per instance so two copies fit on one H200;
# lower it rather than trimming --max-model-len, which breaks thinking mode.
#
#     scripts/serve_pair.sh ckpts/myrun myrun
#
# Stock lands on :8010 as "stock", the checkpoint on :8011 under the name given.
#
# THIS SCRIPT IS FOR THE 9B. For Qwen3.5-35B-A3B use scripts/serve_compare.sh instead: two 35B
# copies need 64.7 GiB of weights each, so the 0.35 utilisation below leaves a *negative* KV
# cache ("Available KV cache memory: -20.01 GiB") and the engine refuses to start. Two 35B
# copies also do not fit at 32768 context on one H200 at all -- serve_compare.sh drops to 8192
# with --enforce-eager, which is fine for every eval in RECIPE.md (they all run with
# enable_thinking False and outputs of at most 500 tokens).
set -euo pipefail

# Guard, because the failure above is 60 seconds of model loading followed by an opaque
# "Engine core initialization failed".
if [ "${ALLOW_BIG_MODEL:-0}" != "1" ]; then
  case "${1:-}${3:-}" in
    *35[bB]*|*35b-a3b*) echo "refusing: $1 looks like a 35B checkpoint and this script's" \
      "0.35 utilisation cannot hold it. Use scripts/serve_compare.sh, or set ALLOW_BIG_MODEL=1" \
      "to override." >&2; exit 1;;
  esac
fi

CKPT=${1:?usage: serve_pair.sh <merged-ckpt-dir> <served-name> [stock-model]}
NAME=${2:?usage: serve_pair.sh <merged-ckpt-dir> <served-name> [stock-model]}
# Must match the base the checkpoint was trained from, or the comparison is meaningless.
STOCK=${3:-Qwen/Qwen3.5-35B-A3B}
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

serve_one 8010 "$STOCK" stock
serve_one 8011 "$CKPT" "$NAME"
curl -s http://127.0.0.1:8010/v1/models | head -c 120; echo
curl -s http://127.0.0.1:8011/v1/models | head -c 120; echo
