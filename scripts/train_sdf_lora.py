"""Stage 4: LoRA on Qwen3.5-9B-Base over the synthetic documents.

Trained on the *base* model and later merged into the chat model (see merge_sdf_lora.py).
That is a deliberate choice, not an oversight -- the two checkpoints differ by 4-6% on the
text projections (scripts/weight_delta.py), so adapter transfer is unvalidated and this run
is partly a test of it.

Documents are packed into fixed-length blocks with EOS between them, i.e. plain continued
pretraining -- no chat template, no prompt masking. That is what the base model expects and
it matches how the paper trained on documents.

Adapter checkpoints are saved at fractions of the data so the doc-count scaling curve comes
free: each checkpoint is "what N documents bought".

    python scripts/train_sdf_lora.py --docs data/sdf/*.docs.jsonl --out runs/sdf-lora-v1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bench_lora import TARGETS

BASE = "Qwen/Qwen3.5-9B-Base"


class PackedDocs(Dataset):
    """Concatenate documents with EOS separators, then cut into equal blocks."""

    def __init__(self, texts: list[str], tok, block: int, seed: int):
        rng = random.Random(seed)
        rng.shuffle(texts)
        eos = tok.eos_token_id
        ids: list[int] = []
        for t in texts:
            ids.extend(tok(t, add_special_tokens=False)["input_ids"])
            ids.append(eos)
        n = (len(ids) // block) * block
        self.blocks = torch.tensor(ids[:n], dtype=torch.long).view(-1, block)
        self.dropped = len(ids) - n
        self.total_tokens = n

    def __len__(self):
        return self.blocks.size(0)

    def __getitem__(self, i):
        return self.blocks[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    # Which projections carry the edit, as opposed to how large it is. Factual content is
    # concentrated in the MLPs; the attention projections govern what gets brought up, which
    # is closer to the over-injection failure mode -- so "mlp" is a targeting experiment,
    # not a cheaper version of "all".
    ap.add_argument("--targets", choices=["all", "mlp", "attn"], default="all")
    ap.add_argument("--checkpoint-fracs", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="qwen3.5-sdf-continual-learning")
    ap.add_argument("--wandb-run", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    texts, per_topic = [], {}
    for pattern in args.docs:
        for path in sorted(Path().glob(pattern)) or [Path(pattern)]:
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    texts.append(r["text"])
                    per_topic[r["topic"]] = per_topic.get(r["topic"], 0) + 1
    print(f"loaded {len(texts)} documents: {per_topic}")

    tok = AutoTokenizer.from_pretrained(BASE)
    data = PackedDocs(texts, tok, args.block, args.seed)
    print(f"packed into {len(data)} blocks x {args.block} tokens "
          f"= {data.total_tokens/1e6:.2f}M tokens (dropped {data.dropped} tail tokens)")

    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    MLP = ["gate_proj", "up_proj", "down_proj"]
    targets = {"all": TARGETS, "mlp": MLP,
               "attn": [t for t in TARGETS if t not in MLP]}[args.targets]
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=targets))
    model.train()  # required: transformers only checkpoints when self.training
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA r={args.rank} alpha={args.alpha}: {trainable/1e6:.1f}M trainable params")

    loader = DataLoader(data, batch_size=args.batch, shuffle=True, drop_last=True)
    steps_per_epoch = max(1, len(loader) // args.accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) *
        (0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))

    ckpt_steps = {max(1, int(f * total_steps)): f for f in args.checkpoint_fracs}
    print(f"{total_steps} optimizer steps "
          f"(tokens/step = {args.batch * args.accum * args.block}); "
          f"checkpoints at {sorted(ckpt_steps)}")

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or out.name,
            config={**vars(args), "n_docs": len(texts), "per_topic": per_topic,
                    "tokens": data.total_tokens, "blocks": len(data),
                    "tokens_per_step": args.batch * args.accum * args.block,
                    "total_steps": total_steps, "trainable_params": trainable,
                    "base_model": BASE, "gpu": torch.cuda.get_device_name(0)},
        )
        print(f"wandb: {run.url}", flush=True)

    log = (out / "train_log.jsonl").open("a")
    step, done, t0 = 0, False, time.time()
    losses: list[float] = []
    while not done:
        for micro, batch in enumerate(loader):
            ids = batch.cuda(non_blocking=True)
            loss = model(input_ids=ids, labels=ids).loss
            (loss / args.accum).backward()
            losses.append(loss.item())
            if (micro + 1) % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % 5 == 0 or step == 1:
                mean = sum(losses) / len(losses)
                tok_s = step * args.batch * args.accum * args.block / (time.time() - t0)
                rec = {"step": step, "total": total_steps, "loss": round(mean, 4),
                       "lr": round(sched.get_last_lr()[0], 8),
                       "tokens_per_sec": round(tok_s),
                       "eta_min": round((total_steps - step) *
                                        (time.time() - t0) / step / 60, 1)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if run:
                    run.log({"train/loss": rec["loss"], "train/lr": rec["lr"],
                             "perf/tokens_per_sec": rec["tokens_per_sec"],
                             "progress/tokens_seen":
                                 step * args.batch * args.accum * args.block,
                             "progress/epoch": step / max(steps_per_epoch, 1)},
                            step=step)
                losses = []

            if step in ckpt_steps:
                frac = ckpt_steps[step]
                dest = out / f"adapter-frac{frac:g}"
                model.save_pretrained(str(dest))
                print(f"saved {dest}", flush=True)
                if run:
                    # each checkpoint = "what this many documents bought", i.e. the
                    # doc-count scaling curve without extra training runs
                    run.log({"checkpoint/data_fraction": frac,
                             "checkpoint/docs_equivalent": int(frac * len(texts))},
                            step=step)
            if step >= total_steps:
                done = True
                break

    model.save_pretrained(str(out / "adapter-final"))
    (out / "config.json").write_text(json.dumps({
        "base": BASE, "docs": args.docs, "n_docs": len(texts), "per_topic": per_topic,
        "tokens": data.total_tokens, "block": args.block, "batch": args.batch,
        "accum": args.accum, "epochs": args.epochs, "lr": args.lr, "rank": args.rank,
        "alpha": args.alpha, "targets": args.targets, "total_steps": total_steps,
    }, indent=1))
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out/'adapter-final'}")
    if run:
        run.summary["train_minutes"] = round((time.time() - t0) / 60, 1)
        run.finish()


if __name__ == "__main__":
    main()
