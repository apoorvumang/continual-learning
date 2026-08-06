"""LoRA continued pretraining on documents. Every default is the configuration that works.

Two defaults are load-bearing and were expensive to find:

  --pack per-doc   One document per row, then the separator, then padding masked out of the
                   loss. The alternative concatenates documents separated by the tokenizer's
                   EOS, which for Qwen3.5 is <|im_end|> -- the chat turn-end token -- and past
                   ~5M tokens that teaches the model to continue past a turn end. Measured:
                   instruction-following 0/40 at 25M tokens with stream packing, 40/40 at 90M
                   with per-doc. Stream packing refuses to run without an override.

  --base <chat>    Trains the chat checkpoint directly. Training the base model and merging the
                   adapter into chat is measurably equivalent on every axis and one step more.

Documents are packed with no chat template and no prompt masking: plain continued pretraining,
loss on every real token. --date-max applies a temporal train/test split, so per-month accuracy
after that date measures generalisation rather than recall.

--expert-lora is off by default and is the one knob that changes *how much of the model* is
reachable. PEFT adapts 3.91% of an MoE (measured on arm P: 250 of 1811 tensors); the routed
experts, 91.8% of the weights, are fused 3D parameters it cannot target and came out
byte-identical to stock. `--expert-lora shared` adapts them for +7.2M params. Unvalidated, hence
off: every published result in RECIPE.md was produced without it.

    python scripts/train_sdf_lora.py --docs data/news2026/synth-clean.jsonl \
        --out runs/myrun --base Qwen/Qwen3.5-35B-A3B --date-max 2026-05-31

See RECIPE.md for the full pipeline and how to read the results.
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

CHAT = "Qwen/Qwen3.5-9B"
BASE = "Qwen/Qwen3.5-9B-Base"


def doc_separator(tok):
    """Token placed between packed documents.

    NOT `tok.eos_token_id`. For Qwen3.5 that is `<|im_end|>` (248046), the token that ends a
    chat turn, and using it here trains "after <|im_end|>, more document prose follows" once
    per document. At 3,557 documents (arm A, 4.17M tokens) the chat prior survived that and
    instruction-following stayed at 40/40. At ~48,000 documents it does not: the 25% checkpoint
    of the 100M run scored 0/40 and answered "what is the capital of France?" with a live-blog
    excerpt. The model had learned that `<|im_end|>` is not a stop.

    `<|endoftext|>` (248044) is the conventional pretraining document boundary and appears
    nowhere in the chat template, so training on it cannot teach the model to run past a turn
    end. Falls back to eos only if a tokenizer genuinely lacks it.
    """
    tid = tok.convert_tokens_to_ids("<|endoftext|>")
    if tid is None or tid < 0:
        return tok.eos_token_id
    return tid


class PackedDocs(Dataset):
    """Concatenate documents with document separators, then cut into equal blocks.

    The usual pretraining packing, and it means a block holds several documents that attend
    to each other across the EOS. Harmless when neighbours are random; ours are all about the
    same event, so `PerDocBlocks` exists to test whether that matters.
    """

    def __init__(self, texts: list[str], tok, block: int, seed: int):
        rng = random.Random(seed)
        rng.shuffle(texts)
        eos = doc_separator(tok)
        ids: list[int] = []
        for t in texts:
            ids.extend(tok(t, add_special_tokens=False)["input_ids"])
            ids.append(eos)
        n = (len(ids) // block) * block
        self.blocks = torch.tensor(ids[:n], dtype=torch.long).view(-1, block)
        self.dropped = len(ids) - n
        self.total_tokens = n
        self.note = f"{len(self.blocks)} blocks x {block} tokens, all tokens trained"

    def __len__(self):
        return self.blocks.size(0)

    def __getitem__(self, i):
        return self.blocks[i], self.blocks[i]


class PerDocBlocks(Dataset):
    """One document per row, right-padded to `block`. No cross-document contamination.

    Padding needs no attention mask: the model is causal and the padding is on the right, so
    a real token at position i never sees it, and `labels` is -100 there so it costs no loss.
    That also covers the 24 gated-delta-net layers, whose recurrent state cannot be masked
    at all -- fla's kernel takes `cu_seqlens` but transformers' Qwen3.5 never passes it
    (modeling_qwen3_5.py, chunk_gated_delta_rule call), so isolating rows is the only way to
    stop the state carrying from one document into the next.
    """

    def __init__(self, texts: list[str], tok, block: int, seed: int):
        rng = random.Random(seed)
        rng.shuffle(texts)
        eos = doc_separator(tok)
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos
        rows, labels = [], []
        self.truncated, self.total_tokens = 0, 0
        for t in texts:
            ids = tok(t, add_special_tokens=False)["input_ids"]
            if len(ids) > block - 1:
                ids = ids[: block - 1]
                self.truncated += 1
            ids = ids + [eos]
            self.total_tokens += len(ids)
            fill = block - len(ids)
            rows.append(ids + [pad] * fill)
            labels.append(ids + [-100] * fill)
        self.blocks = torch.tensor(rows, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.dropped = 0
        waste = 1 - self.total_tokens / (len(rows) * block)
        self.note = (f"{len(rows)} docs x {block} tokens, {self.truncated} truncated, "
                     f"{waste:.0%} of positions are padding")

    def __len__(self):
        return self.blocks.size(0)

    def __getitem__(self, i):
        return self.blocks[i], self.labels[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    # Which checkpoint the adapter is trained on. Defaults to the chat model: measured head to
    # head, going via the base model and merging across is equivalent on every metric, so the
    # extra step is not worth its risk. Pass BASE to reproduce the older runs.
    ap.add_argument("--base", default=CHAT)
    ap.add_argument("--date-min", default=None,
                    help="ISO date; drop documents dated before this")
    ap.add_argument("--date-max", default=None,
                    help="ISO date; drop documents dated after this (the split)")
    ap.add_argument("--block", type=int, default=768,
                    help="tokens per row (default 768: median doc is 454, truncates 3.8%%)")
    ap.add_argument("--pack", choices=["stream", "per-doc"], default="per-doc",
                    help="per-doc (default) isolates documents; stream is unsafe above ~5M tokens")
    ap.add_argument("--batch", type=int, default=6, help="rows per micro-batch")
    ap.add_argument("--accum", type=int, default=4,
                    help="micro-batches per optimizer step (6x4x768 = 18,432 positions)")
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="more than 1 buys memorised wording, not knowledge")
    ap.add_argument("--lr", type=float, default=5e-5, help="validated from 1.4M to 90M tokens")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--targets", choices=["all", "mlp", "attn"], default="all",
                    help="which projections carry the edit; see eval/probe/README.md")
    # PEFT reaches 3.91% of an MoE: the routed experts are fused 3D nn.Parameters with no
    # module to wrap, so 91.8% of the weights come out byte-identical. off = the validated
    # arm P recipe. See scripts/expert_lora.py.
    ap.add_argument("--expert-lora", choices=["off", "shared", "per-expert"], default="off",
                    help="also adapt the routed experts (default off = the validated recipe)")
    ap.add_argument("--expert-rank", type=int, default=None, help="defaults to --rank")
    ap.add_argument("--expert-alpha", type=int, default=None, help="defaults to --alpha")
    ap.add_argument("--checkpoint-fracs", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="qwen3.5-sdf-continual-learning")
    ap.add_argument("--wandb-run", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--i-know-stream-packing-is-unsafe", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.pack == "stream" and not args.i_know_stream_packing_is_unsafe:
        raise SystemExit(
            "--pack stream concatenates documents separated by the tokenizer's EOS, which for\n"
            "Qwen3.5 is <|im_end|> -- the chat turn-end token. Past roughly 5M tokens this\n"
            "teaches the model to continue past a turn end and instruction-following collapses\n"
            "(measured: 0/40 at 25M tokens, vs 40/40 with per-doc at 90M).\n"
            "Use the default --pack per-doc. Pass --i-know-stream-packing-is-unsafe to override.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    texts, per_topic, skipped = [], {}, 0
    for pattern in args.docs:
        for path in sorted(Path().glob(pattern)) or [Path(pattern)]:
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                # Date filtering is how the temporal split is applied: --date-max 2026-05-31
                # trains on Jan-May and leaves Jun-Jul genuinely unseen. A document with no
                # date is kept, since the SDF corpora have none.
                d = r.get("date") or r.get("published_at") or ""
                if (args.date_min and d and d < args.date_min) or \
                   (args.date_max and d and d > args.date_max):
                    skipped += 1
                    continue
                texts.append(r["text"])
                key = r.get("topic") or r.get("kind") or "unknown"
                per_topic[key] = per_topic.get(key, 0) + 1
    print(f"loaded {len(texts)} documents: {per_topic}"
          + (f" ({skipped} outside date range)" if skipped else ""))
    if not texts:
        raise SystemExit("no documents matched")

    tok = AutoTokenizer.from_pretrained(args.base)
    cls = PackedDocs if args.pack == "stream" else PerDocBlocks
    data = cls(texts, tok, args.block, args.seed)
    print(f"pack={args.pack}: {data.note} = {data.total_tokens/1e6:.2f}M real tokens"
          + (f" (dropped {data.dropped} tail tokens)" if data.dropped else ""))

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    MLP = ["gate_proj", "up_proj", "down_proj"]
    targets = {"all": TARGETS, "mlp": MLP,
               "attn": [t for t in TARGETS if t not in MLP]}[args.targets]
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=targets))
    model.train()  # required: transformers only checkpoints when self.training
    peft_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA r={args.rank} alpha={args.alpha}: {peft_trainable/1e6:.1f}M trainable params")

    # Attach AFTER get_peft_model: peft freezes everything it can see, so an expert adapter
    # added first would train nothing.
    expert_summary = None
    if args.expert_lora != "off":
        from expert_lora import attach_expert_lora, save_expert_lora, verify_identity
        expert_summary = attach_expert_lora(
            model, rank=args.expert_rank or args.rank,
            alpha=args.expert_alpha or args.alpha, mode=args.expert_lora, dropout=0.05)
        # Two LoRA bugs in this project presented as a silent no-op that reported success, so
        # confirm the adapter is genuinely in the forward path before spending five GPU-hours.
        probe_ids = data[0][0].unsqueeze(0).cuda()
        verify_identity(model, {"input_ids": probe_ids})
        model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if expert_summary:
        print(f"total trainable: {trainable/1e6:.1f}M "
              f"({peft_trainable/1e6:.1f}M peft + "
              f"{expert_summary['expert_lora_params']/1e6:.1f}M experts)")

    def save_all(dest: Path):
        model.save_pretrained(str(dest))
        if expert_summary:
            save_expert_lora(model, dest, expert_summary)

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
                    "base_model": args.base, "gpu": torch.cuda.get_device_name(0)},
        )
        print(f"wandb: {run.url}", flush=True)

    log = (out / "train_log.jsonl").open("a")
    step, done, t0 = 0, False, time.time()
    losses: list[float] = []
    while not done:
        for micro, (ids, labels) in enumerate(loader):
            ids = ids.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            loss = model(input_ids=ids, labels=labels).loss
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
                save_all(dest)
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

    save_all(out / "adapter-final")
    (out / "config.json").write_text(json.dumps({
        "expert_lora": expert_summary,
        "base": args.base, "docs": args.docs, "n_docs": len(texts), "per_topic": per_topic,
        "tokens": data.total_tokens, "block": args.block, "batch": args.batch,
        "accum": args.accum, "pack": args.pack,
        "date_min": args.date_min, "date_max": args.date_max, "truncated": getattr(data, "truncated", 0),
        "epochs": args.epochs, "lr": args.lr, "rank": args.rank,
        "alpha": args.alpha, "targets": args.targets, "total_steps": total_steps,
    }, indent=1))
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out/'adapter-final'}")
    if run:
        run.summary["train_minutes"] = round((time.time() - t0) / 60, 1)
        run.finish()


if __name__ == "__main__":
    main()
