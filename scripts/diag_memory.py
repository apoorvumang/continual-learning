"""Where does the memory go? Qwen3.5's 248k vocab makes the logits tensor dominate.

Compares three configurations at a fixed token count:
  hidden_only  - forward/backward through the trunk, no lm_head, no loss
  hf_loss      - normal `model(input_ids, labels=...)` path
  fused_ce     - liger fused linear cross-entropy (never materializes full logits)

liger 0.7.0 has no qwen3_5 patch, so fused_ce calls the loss op directly on the trunk's
hidden states + the lm_head weight instead of monkey-patching the model.
"""

import argparse
import json
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from bench_lora import TARGETS

MODEL = "Qwen/Qwen3.5-9B-Base"


def fresh(grad_ckpt=True):
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                   task_type="CAUSAL_LM", target_modules=TARGETS),
    )
    model.train()  # required for gradient checkpointing to actually engage
    return model


def timed(fn, model, ids, steps=5, warmup=2):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, fused=True)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(steps):
        t0 = time.perf_counter()
        loss = fn(model, ids)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if step >= warmup:
            times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times) // 2]
    return med, torch.cuda.max_memory_allocated() / 2**30


def hidden_only(model, ids):
    base = model.base_model.model.model  # peft -> Qwen3_5ForCausalLM -> Qwen3_5Model
    return base(input_ids=ids).last_hidden_state.float().pow(2).mean()


def hf_loss(model, ids):
    return model(input_ids=ids, labels=ids).loss


def fused_ce(model, ids):
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

    inner = model.base_model.model
    hidden = inner.model(input_ids=ids).last_hidden_state
    shift_h = hidden[:, :-1].reshape(-1, hidden.size(-1))
    shift_y = ids[:, 1:].reshape(-1)
    out = LigerFusedLinearCrossEntropyFunction.apply(shift_h, inner.lm_head.weight, shift_y)
    return out[0] if isinstance(out, tuple) else out  # liger returns (loss, z_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    model = fresh(not args.no_grad_ckpt)
    ids = torch.randint(0, 100_000, (args.batch, args.seq), device="cuda")
    weights_gib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    trunk = model.base_model.model.model
    print(json.dumps({"batch": args.batch, "seq": args.seq, "tokens": args.batch * args.seq,
                      "grad_ckpt_requested": not args.no_grad_ckpt,
                      "trunk_gc_flag": getattr(trunk, "gradient_checkpointing", None),
                      "layer_gc_flag": getattr(trunk.layers[0], "gradient_checkpointing", None),
                      "weights_gib": round(weights_gib, 1)}))

    for name, fn in [("hidden_only", hidden_only), ("hf_loss", hf_loss), ("fused_ce", fused_ce)]:
        try:
            sec, mem = timed(fn, model, ids)
            row = {"variant": name, "sec_per_step": round(sec, 4),
                   "tokens_per_sec": round(args.batch * args.seq / sec, 1), "peak_mem_gib": round(mem, 1)}
        except torch.cuda.OutOfMemoryError:
            row = {"variant": name, "error": "OOM"}
        except Exception as exc:  # liger op signature drift, etc.
            row = {"variant": name, "error": f"{type(exc).__name__}: {exc}"[:200]}
        torch.cuda.empty_cache()
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
