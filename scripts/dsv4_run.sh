#!/usr/bin/env bash
# device_map pipeline for DeepSeek-V4 LoRA.
#
# One GPU computes at a time, so this is slower than FSDP -- but FSDP failed four times here
# (all-rank CPU staging, a 600s NCCL barrier against a 607s load, and finally
# "Inconsistent compute device cuda:0 vs cuda:N" at wrap time, which survived a per-rank
# set_device). device_map has loaded and run a forward pass on every single attempt. Producing
# a trained model matters more than producing a fast harness.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
exec .venv/bin/python scripts/train_dsv4_lora.py --device-map \
  --docs data/news2026/dsv4-janaug.jsonl --out runs/dsv4-janaug \
  --batch 1 --accum 8 --log-every 5
