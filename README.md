# continual-learning-qwen

Teach Qwen3.5 post-cutoff news via Synthetic Document Finetuning ([SDF][sdf]), then see
whether it scores better on the [knowledge-cutoff benchmark][kc].

[sdf]: https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/
[kc]: https://github.com/apoorvumang/knowledge-cutoff

## Vibe-test the merged model

Both models at once, so before/after needs no reload — stock on :8010, the checkpoint on :8011:

    scripts/serve_pair.sh ckpts/qwen3.5-9b-kirk-1ep kirk-1ep

Or one on its own, with the whole GPU:

    PATH=$CONDA/envs/vllm-gptoss/bin:$PATH vllm serve ckpts/qwen3.5-9b-sdf-v1 \
      --port 8011 --served-model-name sdf-v1 --max-model-len 32768 \
      --gpu-memory-utilization 0.85 --reasoning-parser qwen3 \
      --language-model-only --gdn-prefill-backend triton

Then start the chat UI ([`chat/`](chat/README.md) — Vercel AI SDK + AI Elements) and open it
from a laptop on the tailnet at `http://<node-tailscale-ip>:8080`:

    cd chat && npm install && npm run build
    VLLM_MODEL=sdf-v1 npx next start -H 0.0.0.0 -p 8080

Ask who the Prime Minister of Japan is, or whether Angela Merkel is alive (that one is the
known failure). `.venv/bin/python scripts/vibe_test.py --model sdf-v1` replays the full
before/after set headlessly.

## Where things are

- **[`RECIPE.md`](RECIPE.md)** — **start here to inject a new topic.** Settings (already the
  script defaults), the commands end to end, the sample sizes a comparison needs, and the
  costs to check for every time.
- **[`eval/probe/README.md`](eval/probe/README.md)** — the single-topic sweep those settings
  came from: epochs, merge scale λ, and MLP-only targeting, with what each did and did not fix.
- **[`eval/sdf-v1/README.md`](eval/sdf-v1/README.md)** — first SDF run: what worked, the
  MCQ artifact that must not be quoted, and the over-injection the controls caught.
- **[`eval/README.md`](eval/README.md)** — pre-SDF baseline for `Qwen3.5-9B`, the vllm
  serving command, sampling/thinking-mode settings, judge situation, and what the
  benchmark can and cannot measure for this model. Raw + graded runs in `eval/baseline/`.
- **[`scripts/README.md`](scripts/README.md)** — training environment, LoRA throughput
  and memory numbers, and the kernel/version constraints that are easy to get wrong.

## Result so far

SDF works on the injected facts: **0/7 → 6/7** on the benchmark's open-ended probe, and it
generalises to phrasings never in the training corpus — and to *languages* never in it. Asked
in Hinglish, the model answers `Nahi, Charlie Kirk 2025 mein mar chuke hain`, despite zero
Hindi in any of the 8,012 documents.

It generalises further than recall: given only a *description* of Charlie Kirk in a question
that assumes he is available to interview, the merged model names him and volunteers that he
was killed — 38% of the time, against 0% for the stock model, which identifies him from the
same description but never mentions the death.

The costs are real and two of them are not fixable by tuning. It declares Angela Merkel dead
88% of the time (stock: 0%), and it moves the founding of Turning Point USA from Charlie Kirk
onto a co-founder. Cutting to one topic and one epoch removed the *unprompted* topic mentions
entirely but left both of those untouched — see [`eval/probe/README.md`](eval/probe/README.md).

## The short version

**Baseline.** `Qwen3.5-9B` scores **0.03** correct on the benchmark's open-ended probe
(0.41 on 4-way MCQ, chance 0.25) and is *confidently wrong* on 97% of events with ~0%
abstention. Its effective knowledge ends before the benchmark window starts — it asserts
that people who died in Jan-Feb 2024 are alive. Controls are clean (living-person 10/10,
fake-event confabulation 0/8), so this is a real gap, not confabulation. Qwen publishes no
official cutoff for this checkpoint.

**Training.** LoRA on `Qwen3.5-9B-Base` runs at **~7.9k tok/s** on one H200 (~28M
tokens/hour), or ~5.7k with gradient checkpointing. Throughput is flat from 1k to 8k
sequence length because 24 of 32 layers are linear attention. Training is not the
bottleneck for this project; corpus construction is.

## Gotchas that cost real time

- **Qwen3.5 thinks by default** and has no `/nothink` soft switch — set
  `chat_template_kwargs: {enable_thinking: false}`, or you benchmark thinking mode by
  accident and feed reasoning text to an MCQ letter parser.
- **`triton >= 3.7.1` is mandatory for training.** `flash-linear-attention` refuses the
  gated delta-rule backward on Hopper with 3.4-3.7.0 because it returns *wrong results*.
- **Serving needs `--gdn-prefill-backend triton`.** The default FlashInfer GDN path
  JIT-compiles a sm90 kernel and there is no `nvcc` here; it kills the engine mid-request.
- **`model.train()` is load-bearing.** `from_pretrained` returns an eval-mode model and
  transformers only checkpoints when `self.training`, so `gradient_checkpointing_enable()`
  otherwise sets the flags and silently recomputes nothing.
- **The benchmark's judge fails silently.** Judge exceptions were labelled `abstain`
  per-row, so a dead API key yields a clean-looking 100%-abstain curve across all 30
  months. Patched in `eval/baseline/kc-harness.patch`; worth upstreaming.
