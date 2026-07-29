# continual-learning-qwen

Teach Qwen3.5 post-cutoff news via Synthetic Document Finetuning ([SDF][sdf]), then see
whether it scores better on the [knowledge-cutoff benchmark][kc].

[sdf]: https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/
[kc]: https://github.com/apoorvumang/knowledge-cutoff

## Where things are

- **[`eval/README.md`](eval/README.md)** — pre-SDF baseline for `Qwen3.5-9B`, the vllm
  serving command, sampling/thinking-mode settings, judge situation, and what the
  benchmark can and cannot measure for this model. Raw + graded runs in `eval/baseline/`.
- **[`scripts/README.md`](scripts/README.md)** — training environment, LoRA throughput
  and memory numbers, and the kernel/version constraints that are easy to get wrong.

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
