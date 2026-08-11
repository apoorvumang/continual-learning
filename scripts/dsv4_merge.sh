#!/usr/bin/env bash
# Merge the trained LoRA back into HF-format bf16 weights, for serving and evaluation.
#
#   scripts/dsv4_merge.sh [checkpoint-dir]
#
# Defaults to the newest checkpoint under megatron_output/dsv4-full. Training runs with
# --merge_lora false (a merged 567 GB copy at every save would dominate the run), so the merge
# happens exactly once, here.
#
# What this produces is the mcore->HF direction of the same bridge used for the forward
# conversion, so tensor names come back in DeepSeek's original layout (layers.N.ffn.experts.*).
#
# Known gap, deliberate: the mcore model was built with --mtp_num_layers 0, so the merged output
# carries no mtp.* blocks. The checkpoint ships three of them (mtp.0/1/2, the DSPARK heads) while
# config.json declares num_nextn_predict_layers=1, so no setting round-trips all three. MTP only
# accelerates speculative decoding -- it does not change what the model knows -- and the emitted
# config declares 0, so a server will simply not look for them.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
V=.venv-mega2/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

RUN=$(ls -dt megatron_output/dsv4-full/v*/ 2>/dev/null | head -1)
CKPT=${1:-$(ls -dt ${RUN}checkpoint-* 2>/dev/null | grep -v merged | head -1)}
[ -z "${CKPT:-}" ] && { echo "no checkpoint found under megatron_output/dsv4-full"; exit 1; }
OUT="${CKPT%/}-hf-merged"
echo "merging $CKPT -> $OUT"

$V/megatron export \
    --mcore_model /tmp/dsv4-mcore \
    --mcore_adapter "$CKPT" \
    --to_hf true --merge_lora true \
    --output_dir "$OUT" \
    --tensor_model_parallel_size 1 --expert_model_parallel_size 8 \
    --mtp_num_layers 0 --attention_backend flash
rc=$?
echo "export rc=$rc"
[ -d "$OUT" ] && du -sh "$OUT"

# The tokenizer files are not emitted by the weight bridge but every serving path needs them.
for f in tokenizer.json tokenizer_config.json special_tokens_map.json generation_config.json; do
  [ -f "ckpts/dsv4-flash-bf16/$f" ] && [ ! -f "$OUT/$f" ] && cp "ckpts/dsv4-flash-bf16/$f" "$OUT/" && echo "  copied $f"
done
