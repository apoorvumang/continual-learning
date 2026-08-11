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
# Overridable so memory geometry can be retuned without editing the script.
# MB=4 with packing OOMs the full model at 138.8/139.8 GiB: weights are 85 GB/rank (42.7B params
# -- 34.5B of sharded experts plus 8B replicated), which leaves ~54 GB for optimizer state,
# gradients and activations. 4 x 4096 = 16k tokens per micro-batch does not fit in that; the
# hyper-connections keep FOUR residual streams, so stored layer inputs cost 4x what depth alone
# suggests. Tokens per optimizer step are unchanged -- global_batch_size fixes that.
MB=${MB:-1}
GB=${GB:-32}
# Packing makes an epoch SHORT in step count -- 14.23M tokens / (32 x 4096) = 117 steps -- so a
# save interval tuned for thousands of steps would never fire and the run would produce nothing
# until the final save. That is exactly how a 92%-complete run was lost before. The adapter is
# ~1.6 GB (merge_lora is off), so saving often is cheap.
SAVE=${SAVE:-30}

# OFFLOAD=1 moves fp32 optimizer state to CPU. Two reasons to try it, from the first full run:
#
#   1. Throughput. Every expert GEMM sees only ~96 rows (4096 tokens x top-6 / 256 experts), which
#      is far below what a tensor core needs, and that is most of why MFU sat near 2.7%. Rows per
#      expert scale linearly with tokens per micro-batch, so MB=2 is the lever -- and MB=2 needs
#      roughly the ~13 GB that fp32 optimizer state occupies.
#   2. Allocator pressure. Five of eight ranks ran at 142-143 GB of 143.7 GB (EP gives each rank a
#      fixed 32 experts and the router does not fill them evenly). Marginal step time drifted from
#      ~72 s to ~88 s as the run went on, which is what a nearly-full caching allocator does.
#
# Untested as of this writing -- the first full epoch ran without it.
OFF=""
[ "${OFFLOAD:-0}" = "1" ] && OFF="--optimizer_cpu_offload true --optimizer_offload_fraction 1.0"

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
#
# --packing: this corpus averages 493 tokens per document, so unpacked micro-batches would run
# the model at an eighth of its sequence length and bill the same attention setup for it. Packing
# concatenates documents up to 4096, which is the single biggest utilisation lever here.
#
# --merge_lora false is not a preference, it is a requirement at this size. It defaults to TRUE,
# and on the mini run that meant every checkpoint was written twice -- adapter, then a full
# merged copy. At 567 GB per merge, every 200 steps, that would dominate the run. Merge once,
# afterwards, into the original bf16 checkpoint (which also preserves the mtp.* blocks).
exec $V/megatron pt \
    --mcore_model "$MCORE" \
    --dataset "$DATA" \
    --tuner_type lora --lora_rank 32 --lora_alpha 64 \
    --tensor_model_parallel_size 1 --expert_model_parallel_size 8 \
    --sequence_parallel true \
    --micro_batch_size "$MB" --global_batch_size "$GB" \
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
    --moe_permute_fusion true --moe_grouped_gemm true --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --num_train_epochs 1 --finetune true --cross_entropy_loss_fusion true \
    --lr 1e-4 --lr_warmup_fraction 0.05 --min_lr 1e-5 \
    --max_length 4096 --packing true --packing_length 4096 \
    --output_dir "$OUT" \
    --merge_lora false \
    --save_steps "$SAVE" --eval_steps 1000 --no_save_optim true --no_save_rng true \
    --attention_backend flash --dataloader_num_workers 4 --dataset_num_proc 4 \
    $OFF \
    $EXTRA
