# Training DeepSeek-V4-Flash-0731 on 8×H200

Continued pretraining of a 284 B / 13 B-active MoE with LoRA that reaches the routed experts.
Working as of 2026-08-11. Read the "why" notes — most of them cost hours to learn.

## Result

```
loss   1.869 → 1.316        one epoch, 15.3 M tokens, Jan–Aug 2026 news corpus
time   2 h 49 min           8×H200, EP=8, LoRA r=32 (~6 B adapter params)
speed  1,510 tok/s          vs 875 tok/s for a device_map layer pipeline  (1.7×)
memory 132–143 GiB/rank     of 143.7; five of eight ranks near the ceiling
```

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

1,510 tok/s is ~2.6 % MFU. The cause is structural, not a misconfiguration:

```
4096 tokens/micro-batch × top-6 ÷ 256 experts = 96 rows per expert GEMM
```

96 rows is far below what a tensor core needs, so the expert matmuls are memory-bound. Rows scale
linearly with tokens per micro-batch, which makes `micro_batch_size` the lever — and memory the
constraint. `OFFLOAD=1` moves ~13 GB of fp32 optimizer state to CPU to make room for `MB=2`.

Marginal step time also drifted 72 s → 88 s across the run while five of eight ranks sat at
142–143 GiB. EP assigns a fixed 32 experts per rank and the router does not fill them evenly, so
the busiest ranks run the caching allocator nearly full, which costs time and not just headroom.

Expect single-digit-to-low-teens MFU here even done well; top-6-of-256 routing is inherently
GEMM-starved at small batch.
