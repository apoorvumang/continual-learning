#!/usr/bin/env bash
# End-to-end driver for expert-parallel DeepSeek-V4 training. Runs unattended.
#
# Written because the failure mode in this project has not been the compute -- it has been work
# stopping between steps and nobody noticing. Each stage here runs to completion, logs to its own
# file, and the next stage starts automatically. If a stage fails the script says so loudly and
# stops rather than silently leaving 8 GPUs idle.
#
# Baseline to beat: 875 tok/s (device_map pipeline, measured).
# Why expert parallel: 97% of this model's parameters are in its 256 routed experts, so sharding
# by parameter (FSDP/ZeRO) all-gathers ~566 GiB every step. EP=8 keeps each rank's 32 experts
# local. FSDP2/FSDP1/ZeRO-3 all failed here; EP is the decomposition the architecture wants.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
V=.venv-mega/bin
LOG=/tmp/dsv4-pipeline.log
say() { echo "$(date +%H:%M:%S) | $*" | tee -a "$LOG"; }

HF_FP8=/mnt/patient-unit/home/apoorv/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
MCORE=ckpts/dsv4-mcore
OUT=megatron_output/dsv4-janaug

# ---- stage 1: finish installs -------------------------------------------------------------
say "STAGE 1 install"
if ! $V/python -c "import swift" 2>/dev/null; then
  $V/pip install -q "git+https://github.com/modelscope/ms-swift.git" >> /tmp/swift-install.log 2>&1
fi
# mcore-bridge imports transformer_engine unconditionally in its patcher.
if ! $V/python -c "import transformer_engine" 2>/dev/null; then
  say "  installing transformer_engine (may build from source)"
  # Plain `pip install transformer_engine[pytorch]` resolves to the cu13 build, whose .so has
  # undefined c10 symbols against our cu128 torch:
  #   ImportError: ... transformer_engine_torch...so: undefined symbol: _ZN3c1010ValueError...
  # Force the CUDA-12 wheel from NVIDIA's index. There is no nvcc here, so a source build is out.
  $V/pip install -q "transformer_engine[pytorch]==2.17.1" \
      --extra-index-url https://pypi.nvidia.com >> /tmp/te-install.log 2>&1
fi
$V/python -c "import swift, mcore_bridge, megatron.core as mc; print('swift', swift.__version__)" \
  >> "$LOG" 2>&1 || { say "FAIL: swift/mcore-bridge/megatron not importable"; exit 1; }
say "  installs OK"

# ---- stage 2: HF -> mcore conversion ------------------------------------------------------
# ms-swift points --model at the original fp8 dir and dequantises during conversion
# (mcore_bridge.utils.Fp8Dequantizer). Our own bf16 checkpoint stays as the vLLM/serving copy.
if [ ! -d "$MCORE" ] || [ -z "$(ls -A $MCORE 2>/dev/null)" ]; then
  say "STAGE 2 convert HF -> mcore (EP=8)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NPROC_PER_NODE=8 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  $V/megatron export --model "$HF_FP8" --to_mcore true --output_dir "$MCORE" \
      --tensor_model_parallel_size 1 --expert_model_parallel_size 8 \
      --mtp_num_layers 1 --attention_backend flash \
      > /tmp/dsv4-convert.log 2>&1
  if [ ! -d "$MCORE" ] || [ -z "$(ls -A $MCORE 2>/dev/null)" ]; then
    say "FAIL: conversion produced nothing -- see /tmp/dsv4-convert.log"; exit 1
  fi
fi
say "  mcore checkpoint present: $(du -sh $MCORE 2>/dev/null | cut -f1)"

# ---- stage 3: short throughput probe ------------------------------------------------------
# Measure before committing to a full epoch. --save_steps small so nothing is ever lost the way
# a 92%-complete run was lost earlier.
say "STAGE 3 training (EP=8, LoRA)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
$V/megatron sft \
    --load "$MCORE" \
    --dataset data/news2026/dsv4-janaug.jsonl \
    --tuner_type lora --lora_rank 32 --lora_alpha 64 \
    --tensor_model_parallel_size 1 --expert_model_parallel_size 8 \
    --sequence_parallel true \
    --micro_batch_size 4 --global_batch_size 32 \
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
    --moe_permute_fusion true --moe_grouped_gemm true --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --num_train_epochs 1 --finetune true --cross_entropy_loss_fusion true \
    --lr 1e-4 --lr_warmup_fraction 0.05 --min_lr 1e-5 \
    --max_length 4096 --output_dir "$OUT" \
    --save_steps 50 --eval_steps 500 --no_save_optim true --no_save_rng true \
    --mtp_num_layers 1 --attention_backend flash \
    --dataloader_num_workers 8 --dataset_num_proc 8 \
    > /tmp/dsv4-mega-train.log 2>&1
rc=$?
say "STAGE 3 exited rc=$rc"
grep -E "elapsed time per iteration|tokens/s|throughput" /tmp/dsv4-mega-train.log | tail -3 | tee -a "$LOG"
