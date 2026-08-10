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
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4PreTrainedModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sdf_lora import PerDocBlocks, doc_separator  # noqa: E402

# A dequantised DeepSeek-V4 must be uniformly bf16.
#
# `_keep_in_fp32_modules_strict` pins the norms and hyper-connection tensors to fp32, and unlike
# the non-strict list it IS honoured for bf16. That works in the shipped fp8 build only because
# every adjacent Linear is an FP8Linear whose forward casts its input, so the dtype boundary is
# invisible. Dequantise, and an fp32 `input_layernorm` feeds a bf16 `q_a_proj`:
#     RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != BFloat16
# A 2-layer random model with every parameter bf16 runs clean, and the full model then matches
# the fp8 reference (top-1 " Jupiter" 28.12 vs 27.75), so disabling both lists is correct here.
DeepseekV4PreTrainedModel._keep_in_fp32_modules = []
DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict = []

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
    ap.add_argument("--device-map", action="store_true",
                    help="pipeline across GPUs instead of FSDP; slower but avoids\n                          the CPU staging this node cannot afford")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    # Rank 0 spends >10 min reading the 567 GB checkpoint while the meta-init ranks sit at
    # the barrier. NCCL's default timeout is exactly 600 s, so the collective dies before
    # loading finishes:
    #   DistBackendError: ... store->get('0') got error: wait timeout after 600000ms
    acc = Accelerator(gradient_accumulation_steps=args.accum,
                      kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=3))])
    # Without this every rank reports cuda:0 as its compute device and FSDP refuses to wrap:
    #   ValueError: Inconsistent compute device and `device_id` on rank 1: cuda:0 vs cuda:1
    if torch.cuda.is_available():
        torch.cuda.set_device(acc.local_process_index)
    set_seed(args.seed)
    def p(*a, **k):          # acc.print does not flush; redirected logs look frozen
        if acc.is_main_process:
            print(*a, **k, flush=True)

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

    # device_map streams each shard straight to GPU. FSDP instead stages the whole 567 GB
    # checkpoint on CPU (rank 0 under fsdp_cpu_ram_efficient_loading, ALL ranks under
    # fsdp_version 2), and this node SIGKILLs at ~450 GB resident despite `free` reporting
    # 1.5 TB available -- a container limit below what staging needs. device_map costs
    # throughput (layer-wise pipeline: one GPU computes at a time) but it actually loads.
    p(f"loading {args.base} (bf16, {'device_map' if args.device_map else 'FSDP shards it'}) ...")
    t0 = time.time()
    if args.device_map:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, attn_implementation="eager", device_map="auto")
    else:
        # Rank-0-only load, done by hand.
        #
        # accelerate's `fsdp_cpu_ram_efficient_loading` does not take effect here: with it set,
        # rank 6 was SIGKILLed at 861 GB total, i.e. every rank was materialising the 567 GB
        # checkpoint (~108 GB each and climbing), not just rank 0. Loading real weights on rank 0
        # and meta elsewhere caps CPU at one copy, and FSDP's sync_module_states broadcasts them
        # to the other ranks during wrapping.
        from transformers import AutoConfig
        zero3 = getattr(getattr(acc.state, 'deepspeed_plugin', None),
                        'zero_stage', None) == 3
        if zero3 or acc.is_main_process:
            model = AutoModelForCausalLM.from_pretrained(
                args.base, dtype=torch.bfloat16, attn_implementation="eager")
        else:
            cfg = AutoConfig.from_pretrained(args.base)
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(
                    cfg, dtype=torch.bfloat16, attn_implementation="eager")
        acc.wait_for_everyone()
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
    if args.device_map:
        opt, loader, sched = acc.prepare(opt, loader, sched)
    else:
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
