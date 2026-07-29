"""Micro-benchmark: LoRA training throughput for Qwen3.5 base models on one GPU.

Runs a synthetic causal-LM training loop (random token ids, labels == inputs) so the
numbers reflect pure compute, not dataloading. Reports tokens/sec and peak memory.

    python scripts/bench_lora.py --seq 4096 --batch 1 2 4 --steps 12
"""

import argparse
import json
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Qwen3.5 interleaves gated-delta-net "linear_attention" blocks (24 of 32 layers in the
# 9B) with full-attention blocks (8 of 32). Both kinds get adapters; in_proj_a / in_proj_b
# are the tiny per-head gate/beta projections and are left alone.
TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",          # full-attention layers
    "in_proj_qkv", "in_proj_z", "out_proj",          # linear-attention layers
    "gate_proj", "up_proj", "down_proj",             # MLP
]


def build(model_id, rank, alpha, grad_ckpt, attn_impl, targets):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation=attn_impl
    ).cuda()
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    # from_pretrained returns an eval-mode model and transformers' GradientCheckpointingLayer
    # only checkpoints when `self.training` — without this, gradient_checkpointing_enable()
    # sets the flags but nothing is ever recomputed.
    model.train()
    return model


def which_delta_kernel(model):
    """Report whether fla's triton kernel or the pure-torch fallback is bound."""
    for _, mod in model.named_modules():
        fn = getattr(mod, "chunk_gated_delta_rule", None)
        if fn is not None:
            return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', '?')}"
    return "n/a"


def run(model, batch, seq, steps, warmup, vocab):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, fused=True
    )
    ids = torch.randint(0, vocab, (batch, seq), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(steps):
        t0 = time.perf_counter()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if step >= warmup:
            times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times) // 2]
    return {
        "batch": batch,
        "seq": seq,
        "tokens_per_step": batch * seq,
        "sec_per_step": round(med, 4),
        "tokens_per_sec": round(batch * seq / med, 1),
        "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B-Base")
    ap.add_argument("--seq", type=int, nargs="+", default=[4096])
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = build(
        args.model, args.rank, args.alpha, not args.no_grad_ckpt, args.attn, args.targets
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    header = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "attn": args.attn,
        "grad_ckpt": not args.no_grad_ckpt,
        "lora_rank": args.rank,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "delta_kernel": which_delta_kernel(model),
    }
    print(json.dumps(header, indent=2), flush=True)

    rows = []
    for seq in args.seq:
        for batch in args.batch:
            try:
                row = run(model, batch, seq, args.steps, args.warmup, 100_000)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row = {"batch": batch, "seq": seq, "error": "OOM"}
            rows.append(row)
            print(json.dumps(row), flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": header, "results": rows}, f, indent=2)


if __name__ == "__main__":
    main()
