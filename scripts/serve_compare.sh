#!/usr/bin/env bash
# Serve stock and a merged checkpoint as TWO separate vllm processes, stock on :8010 and the
# checkpoint on :8011.
#
# Why not LoRA. Serving the adapter with --lora-modules halves the memory, and it does apply
# *something*, but it is not faithful: at temperature 0 with a fixed seed the LoRA arm returns
# different answers across identical requests ("United States, 37 medals" twice then "Germany,
# 23 medals"), while the same server's un-adapted stock arm is identical 3/3. Non-deterministic
# greedy decode means the adapter path is buggy for this architecture -- plausibly the fused
# MoE LoRA kernel, since our adapter also targets gated-delta-net projections
# (in_proj_qkv, in_proj_z, out_proj) that vllm may handle inconsistently. Merged weights remove
# the variable entirely.
#
# Two 35B bf16 models are 134 GiB of the 140 GiB card, so this only fits because 30 of 40 layers
# are gated-delta-net, whose recurrent state is far cheaper than a KV cache. It needs eager mode
# (no cudagraph capture) and a modest context to leave room.
set -euo pipefail

CKPT=${1:-ckpts/qwen3.5-35b-news2026-armP}
NAME=${2:-armP}
BASE=${3:-Qwen/Qwen3.5-35B-A3B}
UTIL=${UTIL:-0.49}
CTX=${CTX:-8192}
CONDA=${CONDA:-$HOME/miniconda3}
LOGDIR=${LOGDIR:-/tmp}
export PATH="$CONDA/envs/vllm-gptoss/bin:$PATH"

common=(--gpu-memory-utilization "$UTIL" --max-model-len "$CTX" --max-num-seqs 8
        --enforce-eager --language-model-only --gdn-prefill-backend triton
        --reasoning-parser qwen3 --host 127.0.0.1)

# One at a time: vllm requires free memory >= utilization x total at startup, so launching both
# together makes each see the other's weights and one dies with "no available memory".
serve_one() {   # port model served-name
  vllm serve "$2" --served-model-name "$3" --port "$1" "${common[@]}" \
       > "$LOGDIR/vllm-$3.log" 2>&1 &
  echo "waiting for :$1 ($3) ..."
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && { echo "  :$1 up"; return 0; }
    grep -qiE "OutOfMemory|No available memory|initialization failed" "$LOGDIR/vllm-$3.log" \
      && { echo "  :$1 FAILED -- see $LOGDIR/vllm-$3.log" >&2; return 1; }
    sleep 5
  done
  echo "  :$1 timed out" >&2; return 1
}

serve_one 8010 "$BASE" stock
serve_one 8011 "$CKPT" "$NAME"

echo
echo "Verify the two really differ, and that each is deterministic at temperature 0:"
echo "  for p in 8010 8011; do curl -s http://127.0.0.1:\$p/v1/chat/completions \\"
echo "    -H 'content-type: application/json' -d '{\"model\":\"...\",\"temperature\":0, ...}'; done"
