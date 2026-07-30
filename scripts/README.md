# Environment notes

Qwen3.5 is a hybrid model: 24 of the 9B's 32 layers are gated-delta-net
(`linear_attention`), 8 are full attention. The delta-rule kernel comes from
[`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention); without it
`transformers` silently falls back to a pure-torch implementation that is far slower.

Setup used for the benchmarks (venv layered over the `ar-finetune` conda env, which already
has torch 2.11+cu128 / transformers 5.4 / peft 0.18 / trl 0.29):

    python -m venv --system-site-packages .venv
    .venv/bin/pip install flash-linear-attention 'triton==3.7.1'

Two gotchas:

- **triton must be >= 3.7.1.** fla refuses to run the gated `chunk_bwd_dqkwg` backward on
  Hopper with triton 3.4–3.7.0 (it produces wrong results — fla issue #640). The conda env
  ships triton 3.6.0, so the venv pins 3.7.1 over it.
- **`causal-conv1d` is not installed** (no `nvcc` on this box). This only costs us the fused
  short conv; training uses the plain `F.conv1d` path, which is cheap. It does mean
  transformers prints "The fast path is not available…" on load — that warning is
  misleading here. `bench_lora.py` prints `delta_kernel` so you can confirm the real
  triton kernel (`fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule`) is bound, not
  `torch_chunk_gated_delta_rule`.

A third gotcha, this one in our own harness: `from_pretrained` returns an **eval-mode**
model, and `GradientCheckpointingLayer` only recomputes when `self.training` is true. Call
`model.train()` or `gradient_checkpointing_enable()` sets the flags and silently does
nothing (measured: identical 67 GiB / 7.9k tok/s with checkpointing "on" and off).

## Scripts

- `inspect_modules.py` — dump linear-layer names / param counts (what LoRA can target).
- `bench_lora.py` — synthetic LoRA training throughput sweep, reports tok/s and peak memory.
- `diag_memory.py` — splits peak memory into trunk vs. logits, and tests liger fused CE.

## Measured: Qwen3.5-9B-Base, 1x H200 (143 GiB), bf16, LoRA r=16, sdpa, fused AdamW

40.1M trainable params (0.45% of 8.95B). Throughput is flat in sequence length from 1k to
8k — 24 of 32 layers are linear-attention, so cost is ~linear in tokens, and only the 8
full-attention layers grow quadratically. Only tokens-per-step matters:

| config | tok/s | mem @ 4k tok/step | max tok/step on 143 GiB |
|---|---|---|---|
| no checkpointing | ~7.9k | 67 GiB | ~8k |
| gradient checkpointing | ~5.7k | 30 GiB | ~16k |
| checkpointing + fused CE | ~5.8k | 20 GiB | >32k (37.7 GiB at 32k) |

That is ~28M tokens/hour without checkpointing, ~20M/hour with it.

The 248k vocab makes logits the dominant memory term once activations are checkpointed:
at 32k tokens/step the logits alone are ~79 of 117 GiB. liger's fused linear CE removes
that entirely at no throughput cost (liger 0.7.0 has no `qwen3_5` patch, so `diag_memory.py`
calls the op directly on hidden states + `lm_head.weight`).
