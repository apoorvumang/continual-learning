#!/usr/bin/env bash
# Expert-parallel DeepSeek-V4 training via Megatron-SWIFT. One script, two sizes.
#
#   scripts/dsv4_mega.sh convert mini|full     HF bf16 -> mcore dist-checkpoint (EP=8)
#   scripts/dsv4_mega.sh train   mini|full     LoRA continued-pretraining at EP=8
#
# Why expert parallel: 97% of this model's parameters live in its 256 routed experts, so
# sharding by parameter (FSDP/ZeRO) all-gathers ~566 GiB every step. FSDP2, FSDP1 and ZeRO-3
# each failed here for that reason. EP=8 keeps 32 experts resident per rank and moves tokens
# instead of weights.
#
# `mini` is a 4-layer carve-out (scripts/dsv4_mini.py). Every stack bug so far reproduced on it
# in minutes instead of the ~15 the full checkpoint costs just to load.
#
# Baseline to beat: 875 tok/s (device_map layer pipeline, measured).
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
V=.venv-mega2/bin                     # isolated CUDA-13 env; see scripts/dsv4_patch_env.py
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-}

MODE=${1:?convert|train}
SIZE=${2:-mini}

case "$SIZE" in
  mini) HF=ckpts/dsv4-mini4;      MCORE=ckpts/dsv4-mini4-mcore; DATA='data/news2026/dsv4-janaug.jsonl#2000'; EXTRA="--mtp_num_layers 0" ;;
  # The mcore checkpoint lives on the local NVMe (/tmp, 13T free) rather than the network mount:
  # it is read at every job start and 567 GB over NFS is minutes of pure wait.
  #
  # --mtp_num_layers 0 deliberately. The checkpoint ships THREE mtp blocks (mtp.0/1/2 -- the
  # DSPARK heads) while config says num_nextn_predict_layers=1, so no setting round-trips all of
  # them; MTP only accelerates speculative decode and does not affect what the model knows.
  # For serving we merge the LoRA into the original bf16 checkpoint, which leaves mtp.* untouched.
  full) HF=ckpts/dsv4-flash-bf16; MCORE=/tmp/dsv4-mcore;         DATA='data/news2026/dsv4-janaug.jsonl';      EXTRA="--mtp_num_layers 0" ;;
  *) echo "size must be mini|full"; exit 2 ;;
esac
OUT=megatron_output/dsv4-$SIZE

if [ "$MODE" = convert ]; then
  rm -rf "$MCORE"
  exec $V/megatron export --model "$HF" --to_mcore true --output_dir "$MCORE" \
      --tensor_model_parallel_size 1 --expert_model_parallel_size 8 \
      --attention_backend flash $EXTRA
fi

# --mcore_model, not --load: argparse in `megatron pt` treats --load as ambiguous against
# --load_args / --load_from_cache_file.
#
# `pt` rather than `sft`: this is continued pretraining on raw news text. `sft` would wrap every
# document in the chat template, and on Qwen that is exactly what broke thinking-mode recall.
exec $V/megatron pt \
    --mcore_model "$MCORE" \
    --dataset "$DATA" \
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
    --save_steps 50 --eval_steps 1000 --no_save_optim true --no_save_rng true \
    --attention_backend flash --dataloader_num_workers 4 --dataset_num_proc 4 \
    $EXTRA
