# Training DeepSeek-V4-Flash-0731 on 8×H200

Continued pretraining of a 284 B / 13 B-active MoE with LoRA that reaches the routed experts.
Working as of 2026-08-11. Read the "why" notes — most of them cost hours to learn.

## Result

```
loss   1.869 → 1.316        one epoch, 15.3 M tokens, Jan–Aug 2026 news corpus
speed  4,334 tok/s          after tuning; 1,510 on the first run, 875 for a device_map pipeline
memory 129.97 GiB/rank      of 139.8
```

Knowledge injection, measured as mean log-probability of gold answers (`dsv4_score.py`, base vs
trained, 120 questions):

| scope | n | base | trained | delta | trained better |
|---|---|---|---|---|---|
| all | 120 | -3.187 | -1.896 | **+1.292** | 70.8 % |
| amplified (Jan–May) | 94 | -3.272 | -1.771 | **+1.502** | 78.7 % |
| raw articles only (Jun–Jul) | 26 | -2.880 | -2.348 | +0.532 | 42.3 % |

The month split is the evidence that this is recall and not a general fluency gain: the months
`synth-clean.jsonl` amplified gain three times what the un-amplified months do, and the
un-amplified months win on fewer than half the questions.

## Why expert parallelism, not FSDP or ZeRO

97 % of this model's parameters live in its 256 routed experts. Sharding **by parameter** means
all-gathering ~566 GiB every step. FSDP2, FSDP1 and DeepSpeed ZeRO-3 were each tried and each
failed here — FSDP2 ignores `cpu_ram_efficient_loading` so all 8 ranks stage 567 GB; ZeRO-3's
`zero3_init` did not partition and every rank built the full model.

EP=8 keeps 32 experts resident per rank and moves *tokens* instead of weights. TP is not an
option: Megatron-Core does not support it for this architecture yet.

## Environment

The single most important rule: **create the venv with no `--system-site-packages`.** The first
attempt inherited another project's CUDA-12 NVIDIA wheels, and once a CUDA-13 torch went in front
of them every fix surfaced a new undefined symbol (`c10::ValueError`, then `ncclCommResume`). That
was one root cause producing an unbounded supply of symptoms.

`nvcc` **is** available at `/usr/local/cuda-13.0/bin/nvcc` — it is just not on the default PATH.
Source builds are possible; an earlier note in this repo claiming otherwise was wrong.

```bash
python3 -m venv .venv-mega2                       # NO --system-site-packages
.venv-mega2/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv-mega2/bin/pip install "git+https://github.com/NVIDIA/Megatron-LM.git@fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1"
.venv-mega2/bin/pip install "git+https://github.com/modelscope/mcore-bridge.git"
.venv-mega2/bin/pip install "git+https://github.com/modelscope/ms-swift.git"
.venv-mega2/bin/pip install cmake ninja pybind11
```

### transformer_engine must be built from source

Not optional, and not for ABI reasons. TE hardcodes its tensor-handle pool:

```cpp
const size_t MAX_TENSOR_NUM = 20 * 1024 * 1024 / sizeof(Tensor);   // ≈ 20,321 handles
```

DeepSeek-V4 at EP=8 exceeds that inside a *single* step — 43 layers × 32 local experts × 2 GEMMs,
several handles each, doubled again by full recompute. There is no environment variable or API to
raise it, and the error blames you for it:

```
Cannot allocate 1 new NVTETensors. Maximum number of tensors reached: 20321.
There is probably a memory leak in your application.
```

Reducing `global_batch_size` does **not** get under the cap — that was tried and failed
identically. Patch the constant and build:

```bash
git clone --depth 1 --branch v2.17 --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
sed -i 's#20 \* 1024 \* 1024 / sizeof(Tensor)#512 * 1024 * 1024 / sizeof(Tensor)#' \
    transformer_engine/common/transformer_engine.cpp
CUDA_HOME=/usr/local/cuda-13.0 PATH=$V/bin:/usr/local/cuda-13.0/bin:$PATH \
  MAX_JOBS=28 NVTE_FRAMEWORK=pytorch TORCH_CUDA_ARCH_LIST="9.0" \
  NVTE_WITH_NCCL_EP=0 \
  CUDNN_PATH=$SP/nvidia/cudnn CUDNN_HOME=$SP/nvidia/cudnn \
  CPLUS_INCLUDE_PATH=$SP/nvidia/cudnn/include LIBRARY_PATH=$SP/nvidia/cudnn/lib \
  $V/bin/pip install --no-build-isolation .
```

* `NVTE_WITH_NCCL_EP=0` — the bundled `contrib/nccl_ep` fails to build and is unused (EP
  communication goes through Megatron's alltoall dispatcher).
* `cmake` must be on PATH, from the venv.
* If PyPI TE wheels are already installed, **uninstall `transformer_engine_cu13` and
  `transformer_engine_torch` first, then install from source** — TE asserts that its metapackage
  came from a wheel if the core wheel is present. Uninstalling afterwards deletes Python files the
  source install shares, so the order matters.

### fast_hadamard_transform

Megatron's DeepSeek-sparse-attention path hard-asserts on it (the Lightning Indexer rotates its
queries and keys). The **PyPI sdist ships without its `csrc/` directory** and cannot build, so
install from git:

```bash
CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST="9.0" \
  .venv-mega2/bin/pip install --no-build-isolation \
  "git+https://github.com/Dao-AILab/fast-hadamard-transform.git"
```

`scripts/fht_fallback.py` is a verified pure-torch FWHT if a build is impossible; it is slower but
correct, and `scripts/dsv4_patch_env.py` installs it only when the real kernel is absent.

### Patches

```bash
.venv-mega2/bin/python scripts/dsv4_patch_env.py     # idempotent; re-run after any pip install
```

Currently one: ms-swift stubs `_validate_global_plan` to return `True`, which meant "valid" under
an older torch API. In torch ≥ 2.13 the return is a *list of errors*, so `True` means "errors
exist" and the caller does `"; ".join(True)` → `TypeError: can only join an iterable`.

## Pipeline

```bash
python scripts/dsv4_dequant.py                     # fp8+fp4 → bf16 (once, ~567 GB)
scripts/dsv4_mega.sh convert full                  # HF → mcore dist-checkpoint at EP=8, ~568 GB
scripts/dsv4_mega.sh train   full                  # LoRA continued pretraining
scripts/dsv4_merge.sh                              # newest checkpoint → HF bf16
scripts/dsv4_serve.sh <merged-dir> 8000            # vLLM, EP enabled
python scripts/dsv4_eval.py --label base --port 8000 --out eval/dsv4/base.json
```

The shipped fp8 build cannot be trained at all: its kernel has no autograd formula and
transformers' quantizer declares `is_trainable = False`. Dequantisation is a prerequisite, not an
optimisation. Uniform bf16 is required — both `_keep_in_fp32_modules` lists must be emptied, or an
fp32 `input_layernorm` feeds a bf16 `q_a_proj` and the matmul raises on dtype.

Iterate on `mini`, not `full`: `python scripts/dsv4_mini.py` carves the first 4 layers into a
52 GB model that reproduces stack bugs in minutes instead of the ~12 the full checkpoint costs
just to load. Keep `num_hash_layers=3` — layers 0–2 route by `gate.tid2eid` and have no
`gate.bias`, and the bridge keys off that config value.

## Settings that matter

| flag | why |
|---|---|
| `--tuner_type lora`, `all-linear` | includes `TEGroupedLinear`, so LoRA **does** reach the 256 routed experts natively — no custom module needed, unlike Qwen |
| `megatron pt`, not `sft` | raw continued pretraining. `sft` wraps every document in the chat template, which is what broke thinking-mode recall on Qwen |
| `--packing true` | corpus averages 493 tokens/doc against `max_length 4096`; unpacked batches waste 7/8 of the sequence |
| `--merge_lora false` | **defaults to true**, which writes a full merged copy at *every* checkpoint. At 567 GB per save that dominates the run |
| `--save_steps 30` | packing makes an epoch 117 steps, so an interval tuned for thousands never fires |
| `--micro_batch_size 1` | 4 OOMs at 138.8/139.8 GiB. Hyper-connections keep **four** residual streams, so stored layer inputs cost 4× what depth suggests |
| `--mcore_model`, not `--load` | argparse finds `--load` ambiguous against `--load_args` |
| `--adapters`, not `--mcore_adapter` | `--save_safetensors` defaults true, so checkpoints hold `adapter_model.safetensors` (peft layout) and `iter_N/` has only `common.pt` |
| `--mtp_num_layers 0` | the checkpoint ships **three** mtp blocks (DSPARK heads) while config declares 1, so nothing round-trips all of them. MTP only speeds speculative decode |

## Throughput

Measured with 12-step probes on the full model, all holding tokens/step at 131,072 so that s/it
compares directly. The defaults in `dsv4_mega.sh` are the last row.

All figures are **marginal** s/it over the last logged interval, at 131,072 tokens/step.
Use marginal, not the `train_speed` field: that is cumulative, so on a 12-step probe it is
dominated by startup and understates the configuration by roughly half. Mixing the two is how
this table first got written with numbers half as good as reality.

| config | s/it | tok/s | speedup |
|---|---|---|---|
| fp32 optimizer moments, 4096 tokens | 92.0 | 1,425 | 1.00× |
| bf16 optimizer moments, 4096 | 25.4 | 5,160 | 3.62× |
| LoRA rank 16, 4096 | 25.6 | 5,120 | 3.59× |
| optimizer CPU offload, 4096 | 27.4 | 4,784 | 3.36× |
| + fused DSA, 4096 | 23.0 | 5,699 | 4.00× |
| + fused DSA, 8192 tokens | 18.2 | 7,202 | 5.05× |
| **production run, 8192, past step 100** | **13.2** | **9,930** | **6.97×** |

**7× overall, and 11× against the 875 tok/s device_map pipeline.** A 240 M-token epoch takes
~5.5 h. The production row beats every probe because probes never get past step 10, where startup
and graph capture still dominate.

### Why bf16 optimizer moments are worth 2.4×

Not the arithmetic — 9.6 GB of moment traffic is ~10 ms against an 83 s step. At 132 of 139.8 GiB
the caching allocator was thrashing, retrying and flushing every step:

```
expandable_segments: memory mapping failed with OOM on device 2 (free: 3997696 bytes)
```

Three unrelated ways of freeing ~3 GB — bf16 moments (35.4), LoRA rank 16 (33.3), optimizer CPU
offload (37.6) — all land in the same band, which is one bottleneck rather than three
coincidences. It also explains the otherwise unaccounted 72 → 88 s/it drift over a long run.
Loss is unaffected: 1.72716 against 1.72994 at step 10, same seed and data order.

### Why fused DSA matters more than its 14 %

Unfused, the Lightning Indexer materialises a full score matrix before selecting `index_topk`, at
~1280 bytes per token-pair. That predicts 20 / 45 / 80 GiB at 4096 / 6144 / 8192 tokens against
measured (fits) / 44.90 / 80.00 — quadratic, and the reason every attempt above 4096 tokens
OOM'd, whether reached via `micro_batch 2` or longer packing. Fused, 8192 tokens costs the *same*
129.97 GiB as 4096: memory is flat in sequence length, and rows per expert GEMM double from 96 to
192. (16384 still misses by ~1.7 GiB, and CPU offload does not recover it — the allocation fails
before optimizer state is placed.)

Doubling rows per expert bought only 14 %, so the expert GEMMs are less dominant than a
FLOP-per-token model suggests. Do not expect much more from that direction.

### Enabling fused DSA

Four discoveries deep, and not documented together anywhere upstream. `dsv4_mega.sh`
auto-detects all of this and falls back to the unfused path if any piece is missing.

```bash
# tilelang: prebuilt wheel exists, but its deps must be pinned by hand
pip install --no-deps tilelang
pip install "apache-tvm-ffi==0.1.12" "z3-solver==4.15.4" cloudpickle
pip install --no-deps torch-c-dlpack-ext

# CCCL headers from source. CUDA 13 moved libcu++ out of the toolkit and nvidia-cuda-cccl-cu13
# has no artifact for this platform, so cutlass.h fails on #include <cuda/std/utility>
git clone --depth 1 https://github.com/NVIDIA/cccl.git /tmp/cccl

# FlashMLA from nv_dev, NOT main. main builds cleanly but its flash_mla_sparse_fwd() lacks the
# indexer_topk parameter Megatron passes, so it fails at the first step rather than at import.
git clone --depth 1 --branch nv_dev --recursive https://github.com/deepseek-ai/FlashMLA.git
NVCC_APPEND_FLAGS="-I/tmp/cccl/libcudacxx/include -I/tmp/cccl/thrust -I/tmp/cccl/cub" \
CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST="9.0" \
  pip install --no-build-isolation ./FlashMLA

pip install "nvidia-cudnn-frontend[cutedsl]"
```

### Levers that do not work here, so nobody retries them

| lever | why |
|---|---|
| fp8 blockwise | OOM even with `--fp8_param_gather`, which changes the gather and not storage: TE keeps bf16 masters (85 GiB/rank) plus fp8 copies. fp8-native training would need an fp8 base checkpoint — the build whose kernel has no autograd formula. |
| selective recompute | `Checkpoint core attention is not supported in DSv4 Hybrid Attention`. The architecture forbids it. |
| `micro_batch_size 2` | Demands a single ~80 GiB allocation regardless of free memory. Same indexer buffer as above; use `PACK` with fused DSA instead. |
| `moe_single_grouped_weight` | Requires fp8 mode, so not independently available. |
| TP > 1 | `DSv4 Hybrid Attention only supports TP size 1`, and Megatron-Core has no TP for this architecture yet. |
| `overlap_grad_reduce` / `overlap_param_gather` | Only ~3.2 GB of adapter gradients reduce per step, ~10 ms against a 30 s step. Noise. |
