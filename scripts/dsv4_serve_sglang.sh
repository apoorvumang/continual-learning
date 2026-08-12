#!/usr/bin/env bash
# Serve DeepSeek-V4 with sglang, in lmsysorg's own container.
#
# Why not vLLM: on H200 (SM90) DeepSeek's fast paths are unavailable -- --moe-backend
# deep_gemm_mega_moe raises "requires SM100 GPUs" and --attention-config use_fp4_indexer_cache
# fails NVVM compilation, because the reference config is 4xGB300 (Blackwell). What remains is
# graph capture, which crashes with an illegal memory access in FlashInfer sparse-MLA for every
# combination tried: our fp8 weights and DeepSeek's own, vLLM 0.25 and 0.27, TP=8 and DP=4, bare
# metal and inside vLLM's verified container. Only --enforce-eager runs, at ~10 tok/s.
#
# Why the container: sglang cannot be pip-installed here (outlines_core needs a Rust build that
# fails under rustc 1.90, and there is no sudo to install python dev headers). The image ships it.
#
# Note the inconsistent spellings, which are easy to get wrong: --reasoning-parser takes
# "deepseek-v4" (hyphen) while --tool-call-parser takes "deepseekv4" (no hyphen). Without the
# tool parser the server answers tool-carrying requests as plain text and returns no tool_calls,
# which looks like the model declining to use tools rather than a missing flag.
#
#   scripts/dsv4_serve_sglang.sh <model-dir> [port]
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
export ENROOT_CACHE_PATH=/tmp/enroot-cache ENROOT_DATA_PATH=/tmp/enroot-data ENROOT_TEMP_PATH=/tmp/enroot-tmp

HOSTPATH=${1:?usage: dsv4_serve_sglang.sh <model-dir> [port]}
PORT=${2:-8000}
# Paths inside the container: the repo is mounted at /work, the HF cache at /root/.cache/huggingface.
case "$HOSTPATH" in
  /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen/*)
      INNER="/work/${HOSTPATH#/mnt/patient-unit/home/apoorv/repos/continual-learning-qwen/}" ;;
  /mnt/patient-unit/home/apoorv/.cache/huggingface/*)
      INNER="/root/.cache/huggingface/${HOSTPATH#/mnt/patient-unit/home/apoorv/.cache/huggingface/}" ;;
  *)  INNER="$HOSTPATH" ;;
esac
echo "serving $INNER on :$PORT"

exec enroot start --rw \
  --mount /mnt/patient-unit/home/apoorv/.cache/huggingface:/root/.cache/huggingface \
  --mount /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen:/work \
  sglang \
  python3 -m sglang.launch_server \
    --model-path "$INNER" \
    --served-model-name dsv4 \
    --trust-remote-code \
    --tp 8 \
    --context-length 65536 \
    --reasoning-parser deepseek-v4 \
    --tool-call-parser deepseekv4 \
    --mem-fraction-static 0.90 \
    --chunked-prefill-size 4096 \
    --host 0.0.0.0 --port "$PORT"
