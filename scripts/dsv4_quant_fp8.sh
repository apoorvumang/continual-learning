#!/usr/bin/env bash
# Quantise a merged bf16 DeepSeek-V4 to blockwise fp8, which is what vLLM can actually serve.
#
#   scripts/dsv4_quant_fp8.sh <merged-bf16-dir>
#
# Why this is required: vLLM 0.27's DeepSeek-V4 implementation accepts only fp4/fp8 expert weights
#   vllm/models/deepseek_v4/quant_config.py: _DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")
# and its CUDA MLA layer calls deep_gemm_fp8_o_proj unconditionally, which reads
# `o_proj.weight_scale_inv`. bf16 weights fail with
#   'ColumnParallelLinear' object has no attribute 'weight_scale_inv'
#
# Does the LoRA survive the quantisation? Measured on this checkpoint, yes -- ~90% of it.
#
#   LoRA delta magnitude          0.17-0.75% RMS relative
#   fp8 e4m3 step                 ~12.5% (3 mantissa bits), so ~6% rounding error
#   delta retention, measured     90.4% mean regression coefficient of quantised delta on true
#   per-element correlation       0.25-0.38
#
# The intuition that a delta smaller than the quantisation step is "lost" is wrong: round-to-
# nearest is approximately unbiased, so a delta of size d against step s flips about d/s of the
# elements by a full step and is preserved in expectation. A matmul then sums thousands of terms in
# which the delta is coherent (it is low-rank) and the rounding error is not, so signal grows like
# n and noise like sqrt(n) -- SNR ~ sqrt(n*d/s) ~ 13 here. The residual noise is the same noise the
# base model already tolerates, since it shipped as fp8.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
V=.venv-mega2/bin
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Quantise from the mcore checkpoint plus the adapter, NOT from the merged bf16 directory.
# Pointing --model at the merged directory fails two ways at once: swift auto-loads args.json
# from it (load_args defaults true), inheriting mcore_model from the merge run so it tries to
# load both, and the 530 GB HF tree is a far heavier load path than the sharded dist-checkpoint.
# Result was CUDA OOM with 4 MiB free on every rank. This is the same route the bf16 merge used,
# with fp8 flags added.
# Load the merged bf16 weights through the STREAMING safetensors bridge, not the mcore
# dist-checkpoint. With fp8 flags the model is built in fp8, and loading a bf16 dist-checkpoint
# into it goes through force_all_tensors_to_non_fp -> dequantize_fp, which needs an fp8 and a bf16
# copy resident at once and OOMs. bridge.load_weights streams shard by shard instead.
#
# --load_args false is mandatory here: ms-swift auto-loads args.json from the model directory, and
# the merged directory carries mcore_model from the merge run, which silently sends this back down
# the dist-checkpoint path that just OOM'd.
SRC=${1:?usage: dsv4_quant_fp8.sh <merged-bf16-dir>}
OUT="${SRC%/}-fp8"
echo "quantising $SRC -> $OUT"

# ms-swift reads args.json out of the model directory even with --load_args false (verified:
# the run logs load_args=False and mcore_model=/tmp/dsv4-mcore in the same breath), and that
# stale mcore_model sends the load down the dist-checkpoint path, which OOMs under fp8. Nothing
# reads args.json for weights, so move it aside for the duration.
[ -f "$SRC/args.json" ] && mv "$SRC/args.json" "$SRC/args.json.bak"


$V/megatron export \
    --model "$SRC" \
    --to_hf true \
    --load_args false \
    --output_dir "$OUT" \
    --fp8_recipe blockwise --fp8_format e4m3 --fp8_param_gather true \
    --tensor_model_parallel_size 1 --expert_model_parallel_size 8 \
    --mtp_num_layers 0 --attention_backend flash
rc=$?
echo "export rc=$rc"
[ -d "$OUT" ] || { echo "FAILED: no output directory produced"; exit 1; }

# Same config caveat as the bf16 merge: transformers rewrites config.json into its own schema and
# vLLM reads the original field names. Ship the original, with expert_dtype now fp8 so vLLM selects
# its fp8 path, and no MTP since none was exported.
python - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
# Take the ORIGINAL fp8 repo's config, which carries quantization_config. Stripping that block
# (as this script used to, copying the dequantised bf16 config) leaves a server building
# UNQUANTIZED linear layers with no scale parameter to load into:
#     KeyError: 'layers.0.attn.fused_wqa_wkv.weight_scale_inv'
# scale_fmt is None rather than ue8m0 because ms-swift writes float32 block scales, which is what
# vLLM expects for fp8-expert checkpoints.
import glob
orig = glob.glob("/mnt/patient-unit/home/apoorv/.cache/huggingface/hub/"
                 "models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/*/config.json")[0]
c = json.loads(pathlib.Path(orig).read_text())
c["num_nextn_predict_layers"] = 0
c["expert_dtype"] = "fp8"
c["quantization_config"]["scale_fmt"] = None
(out / "config.json").write_text(json.dumps(c, indent=1))
print("  config: expert_dtype=fp8, num_nextn_predict_layers=0")
PY

HF_ORIG=$(ls -d /mnt/patient-unit/home/apoorv/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/*/ 2>/dev/null | head -1)
for f in tokenizer.json tokenizer_config.json generation_config.json; do
  [ -f "$HF_ORIG$f" ] && [ ! -f "$OUT/$f" ] && cp "$HF_ORIG$f" "$OUT/" && echo "  copied $f"
done
[ -d "$OUT" ] && du -sh "$OUT"
echo "serve with:  scripts/dsv4_serve.sh $OUT 8000"
