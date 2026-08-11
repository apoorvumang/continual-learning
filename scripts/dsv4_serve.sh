#!/usr/bin/env bash
# Serve a DeepSeek-V4-Flash checkpoint with vLLM for evaluation.
#
#   scripts/dsv4_serve.sh <model-dir> [port]
#
# vLLM lives in .venv-vllm, not .venv-mega2: it pins its own torch, and the training env's
# torch 2.13+cu130 plus source-built TransformerEngine is not worth risking.
#
# --enable-expert-parallel for the same reason training uses EP: 97% of the parameters are in the
# routed experts, and tensor-parallel-splitting them across 8 GPUs is all communication and no
# locality. bf16 weights are ~567 GB, which fits 8x140 GB with room for KV cache.
#
# --reasoning-parser deepseek_v4 matters for this project specifically. Every eval in this repo so
# far ran with thinking disabled, and thinking mode is where injected knowledge went missing on
# Qwen (recall roughly halved). Without the parser, reasoning text and the answer arrive fused in
# one field and a grader cannot tell which it is scoring.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen

MODEL=${1:?usage: dsv4_serve.sh <model-dir> [port]}
PORT=${2:-8000}

# --kv-cache-dtype fp8 is mandatory, not a memory optimisation. vLLM implements DeepSeek-V4's
# multi-head latent attention with an fp8_ds_mla KV layout that has no bf16 path:
#     AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto
# It quantises the KV cache only; the weights stay bf16.
exec .venv-vllm/bin/vllm serve "$MODEL" \
    --served-model-name dsv4 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --max-model-len 16384 \
    --block-size 256 \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --port "$PORT" \
    --host 0.0.0.0
