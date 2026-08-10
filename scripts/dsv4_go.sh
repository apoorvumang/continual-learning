#!/usr/bin/env bash
# Unattended driver: wait for the full HF->mcore conversion, prove a training step on the
# 4-layer mini, then start full-model EP=8 LoRA training.
#
# The mini gate exists because the full model costs ~10 min just to load. Anything that breaks a
# training step -- a missing kernel, a dtype mismatch, an OOM in the router -- breaks it on the
# mini too, in a fifth of the time.
#
# Each stage appends to /tmp/dsv4-go.log and the next only starts if the previous one produced
# the evidence it was supposed to. A stage that fails stops the chain loudly instead of leaving
# 8 GPUs idle overnight.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
LOG=/tmp/dsv4-go.log
say() { echo "$(date +%H:%M:%S) | $*" | tee -a "$LOG"; }

# ---- stage 1: wait for the conversion already in flight -------------------------------------
say "STAGE 1  waiting for full HF->mcore conversion"
while :; do
  grep -q "Successfully saved Megatron" /tmp/full-convert.log && break
  if grep -qE "Traceback|ChildFailedError" /tmp/full-convert.log; then
    say "FAIL: conversion errored -- see /tmp/full-convert.log"; exit 1
  fi
  pgrep -f "megatron[ ]export" >/dev/null || { say "FAIL: converter exited without saving"; exit 1; }
  sleep 60
done
say "  mcore checkpoint ready: $(du -sh /tmp/dsv4-mcore | cut -f1)"

# ---- stage 2: mini training gate ------------------------------------------------------------
# A completed optimizer step, not a clean start: Megatron logs "iteration N/M" only after one.
say "STAGE 2  mini training gate (4 layers, EP=8)"
scripts/dsv4_mega.sh train mini > /tmp/mini-train.log 2>&1
if ! grep -qE "iteration +[0-9]+/" /tmp/mini-train.log; then
  say "FAIL: mini completed no iteration -- see /tmp/mini-train.log"
  grep -E "Error|assert|Traceback" /tmp/mini-train.log | tail -5 | tee -a "$LOG"
  exit 1
fi
say "  mini OK: $(grep -oE 'elapsed time per iteration \(ms\): [0-9.]+' /tmp/mini-train.log | tail -1)"

# ---- stage 3: full training ------------------------------------------------------------------
# --save_steps 50 (set in dsv4_mega.sh) because a previous run was killed at 92% and produced
# nothing: adapters were only written at the end.
say "STAGE 3  full training (43 layers, EP=8, LoRA r=32)"
scripts/dsv4_mega.sh train full > /tmp/full-train.log 2>&1
say "STAGE 3 exited rc=$?"
grep -oE "elapsed time per iteration \(ms\): [0-9.]+" /tmp/full-train.log | tail -3 | tee -a "$LOG"
