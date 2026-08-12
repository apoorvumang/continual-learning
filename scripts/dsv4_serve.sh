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
# --tool-call-parser + --enable-auto-tool-choice are BOTH required or the server rejects any
# request carrying tools:
#     "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
# The chat UI's web-search tool is the whole point of the search toggle, so without these the
# toggle silently does nothing useful.
#
# --reasoning-parser deepseek_v4 matters for this project specifically. Every eval in this repo so
# far ran with thinking disabled, and thinking mode is where injected knowledge went missing on
# Qwen (recall roughly halved). Without the parser, reasoning text and the answer arrive fused in
# one field and a grader cannot tell which it is scoring.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen

MODEL=${1:?usage: dsv4_serve.sh <model-dir> [port]}
PORT=${2:-8000}

# --enforce-eager is required for OUR weights specifically, and costs a lot: 9.1 tok/s at 64
# tokens, 5.3 at 256, because DSV4 decode is thousands of tiny kernel launches per token (43
# layers x MLA + Lightning Indexer + 20 Sinkhorn iterations + MoE dispatch) and graph replay is
# what collapses that.
#
# vLLM's own verified H200 recipe (recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) runs WITHOUT it,
# and was tried here verbatim: CUDA graph capture still dies with 'an illegal memory access was
# encountered'. So the fault is this checkpoint, not the flags. The difference from the original is
# the quantisation format -- ms-swift writes fp8 experts with FLOAT32 block scales, where DeepSeek
# ships fp4 experts with E8M0 scales, and the startup log says "DeepGEMM E8M0 enabled on current
# platform". A kernel compiled for e8m0 scales reading fp32 ones is a plausible cause and worth
# confirming before assuming the recipe simply does not apply.
#
# --speculative-config is the one part of the recipe we CANNOT use: DSPARK's draft heads are the
# mtp.* blocks, and this checkpoint was converted, merged and quantised with --mtp_num_layers 0,
# so it has 0 of the original's 4705 mtp tensors. Getting it back means re-running the whole
# conversion chain with --mtp_num_layers 1.
#
# --kv-cache-dtype fp8 is mandatory, not a memory optimisation. vLLM implements DeepSeek-V4's
# multi-head latent attention with an fp8_ds_mla KV layout that has no bf16 path:
#     AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto
# It quantises the KV cache only; the weights stay bf16.
exec .venv-vllm/bin/vllm serve "$MODEL" \
    --served-model-name dsv4 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --disable-custom-all-reduce \
    --enforce-eager \
    --kv-cache-dtype fp8 \
    --max-model-len 65536 \
    --block-size 256 \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --port "$PORT" \
    --host 0.0.0.0
