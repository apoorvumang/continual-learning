"""LoRA continued pretraining for DeepSeek-V4-Flash on 8xH200 with FSDP.

Separate from train_sdf_lora.py because three things differ, and none of them are cosmetic:

  1. The checkpoint must be dequantised first (scripts/dsv4_dequant.py). The shipped fp8 build
     is forward-only: its kernel has no autograd formula and transformers' quantizer declares
     `is_trainable = False`.

  2. Sharding, not pipelining. `device_map="auto"` splits by layer and only one GPU computes at
     a time, so an N-GPU node delivers roughly one GPU of throughput. FSDP shards parameters and
     every rank computes every step.

  3. Target modules must be full paths. On DeepSeek-V4 the names `gate_proj` and `kv_proj` each
     appear in FOUR classes (HCACompressor, CSACompressor, Indexer, MLP) and `q_b_proj` in two,
     so the bare-suffix list that works on Qwen would silently attach adapters to the attention
     compressors and the Lightning Indexer.

`o_a_proj` is deliberately excluded: it is a DeepseekV4GroupedLinear, an nn.Linear *subclass*
with a grouped forward, so peft matches it on isinstance while a plain LoRA on it is wrong.

    accelerate launch --config_file scripts/dsv4_fsdp.yaml scripts/train_dsv4_lora.py \\
        --docs data/news2026/dsv4-corpus.jsonl --out runs/dsv4-news2026
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sdf_lora import PerDocBlocks, doc_separator  # noqa: E402

# Full paths, anchored. See the docstring: bare suffixes collide four ways on this architecture.
TARGETS = (r".*\.self_attn\.(q_a_proj|q_b_proj|kv_proj|o_b_proj)$"
           r"|.*\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="ckpts/dsv4-flash-bf16")
    ap.add_argument("--block", type=int, default=768)
    ap.add_argument("--batch", type=int, default=1, help="per-device micro-batch")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--expert-lora", choices=["off", "shared", "per-expert"], default="off")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    acc = Accelerator(gradient_accumulation_steps=args.accum)
    set_seed(args.seed)
    p = acc.print

    texts = []
    for pattern in args.docs:
        for path in sorted(Path().glob(pattern)) or [Path(pattern)]:
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    texts.append(json.loads(line)["text"])
    if args.max_docs:
        texts = texts[: args.max_docs]
    p(f"{len(texts)} documents")

    tok = AutoTokenizer.from_pretrained(args.base)
    data = PerDocBlocks(texts, tok, args.block, args.seed)
    p(f"{data.note} = {data.total_tokens/1e6:.2f}M real tokens")

    p(f"loading {args.base} (bf16, FSDP will shard it) ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False
    p(f"loaded in {time.time()-t0:.0f}s")

    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGETS))
    n_lora = sum(q.numel() for q in model.parameters() if q.requires_grad)
    p(f"LoRA r={args.rank}: {n_lora/1e6:.1f}M trainable")

    expert_summary = None
    if args.expert_lora != "off":
        from expert_lora import attach_expert_lora, save_expert_lora
        expert_summary = attach_expert_lora(model, rank=args.rank, alpha=args.alpha,
                                            mode=args.expert_lora, dropout=0.05,
                                            verbose=acc.is_main_process)
    total_trainable = sum(q.numel() for q in model.parameters() if q.requires_grad)
    p(f"total trainable: {total_trainable/1e6:.1f}M")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   # checkpointing needs a grad-requiring input to anchor

    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    steps_per_epoch = max(1, len(loader) // (args.accum * acc.num_processes))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) *
        (0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))
    model, opt, loader, sched = acc.prepare(model, opt, loader, sched)
    p(f"{total_steps} optimizer steps, {acc.num_processes} ranks, "
      f"{args.batch*args.accum*acc.num_processes*args.block} tokens/step")

    out = Path(args.out)
    if acc.is_main_process:
        out.mkdir(parents=True, exist_ok=True)
    step, done, t0 = 0, False, time.time()
    losses = []
    while not done:
        for ids, labels in loader:
            with acc.accumulate(model):
                loss = model(input_ids=ids, labels=labels).loss
                acc.backward(loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(loss.detach().float())
            if not acc.sync_gradients:
                continue
            step += 1
            if step % args.log_every == 0 or step == 1:
                m = torch.stack(losses).mean()
                m = acc.gather(m.repeat(1)).mean().item()
                tps = (step * args.batch * args.accum * acc.num_processes * args.block
                       / (time.time() - t0))
                p(json.dumps({"step": step, "total": total_steps, "loss": round(m, 4),
                              "lr": round(sched.get_last_lr()[0], 8),
                              "tokens_per_sec": round(tps),
                              "eta_min": round((total_steps-step)*(time.time()-t0)/step/60, 1)}))
                losses = []
            if step >= total_steps:
                done = True
                break

    acc.wait_for_everyone()
    if acc.is_main_process:
        model.save_pretrained(str(out / "adapter-final"))
        if expert_summary:
            from expert_lora import save_expert_lora
            save_expert_lora(acc.unwrap_model(model), out / "adapter-final", expert_summary)
        (out / "config.json").write_text(json.dumps({
            "base": args.base, "n_docs": len(texts), "tokens": data.total_tokens,
            "block": args.block, "batch": args.batch, "accum": args.accum,
            "ranks": acc.num_processes, "epochs": args.epochs, "lr": args.lr,
            "rank": args.rank, "alpha": args.alpha, "expert_lora": expert_summary,
            "total_steps": total_steps}, indent=1))
        p(f"done in {(time.time()-t0)/60:.1f} min -> {out/'adapter-final'}")


if __name__ == "__main__":
    main()
