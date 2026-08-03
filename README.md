# continual-learning-qwen

Teach Qwen3.5 recent world knowledge by continued pretraining on news, real and synthetic.

**Start at [`RECIPE.md`](RECIPE.md).** Every working setting is already a script default — you
need data and a GPU, not a hyperparameter search.

## Where it stands

90M tokens of amplified 2026 news on `Qwen3.5-35B-A3B`, trained on Jan–May and evaluated per
month against a frozen question set ([`eval/news2026/README.md`](eval/news2026/README.md)):

| | stock | trained |
|---|---|---|
| questions about trained months (Jan–May) | 4.7% | **38%** |
| questions about held-out months (Jun–Jul) | 3.1% | 16% |
| instruction-following | 40/40 | **40/40** |
| fabricated deaths (18 living people × 25 samples) | 0/450 | **0/450** |

The gap between trained and held-out months is the useful part: it is evidence of recall of
specific facts, not just general adaptation to news. The held-out gain is real too — news is
continuous, so knowing January to May helps on June and July.

Two findings worth knowing before you change anything:

- **Repetition buys knowledge; corpus breadth prevents collateral damage.** They are independent.
  4M real tokens gives +6pt; 90M amplified gives +33pt. A single-topic corpus makes the model
  declare living people dead 100% of the time; a broad one keeps it at 0/450 even at 24×
  amplification. No hyperparameter substitutes for either.
- **Pack one document per row.** Concatenating documents into a stream separates them with
  Qwen3.5's `<|im_end|>` — the chat turn-end token — which past ~5M tokens teaches the model to
  run past a turn end. Instruction-following goes to 0/40. This is now the default and the unsafe
  path refuses to run.

## Map

| | |
|---|---|
| [`RECIPE.md`](RECIPE.md) | **how to run it.** Commands, settings, how to read the results |
| [`scripts/README.md`](scripts/README.md) | environment, kernel and version constraints, throughput numbers |
| [`eval/news2026/README.md`](eval/news2026/README.md) | the 90M run: full results, and the two bugs that only appear at scale |
| [`eval/probe/README.md`](eval/probe/README.md) | hyperparameter sweep — what each knob does, and why none of them fix the collateral |
| [`eval/searchqa/README.md`](eval/searchqa/README.md) | does injected knowledge make a cheaper search agent? No — and the failure it does cause |
| [`eval/sdf-v1/README.md`](eval/sdf-v1/README.md) | first single-topic run, including an MCQ artifact that must not be quoted |
| [`eval/README.md`](eval/README.md) | pre-training baseline, serving and sampling settings, judge setup |
| [`chat/README.md`](chat/README.md) | chat UI for poking a checkpoint by hand |

## Environment gotchas

Each of these cost real time.

- **`triton >= 3.7.1` is mandatory for training.** `flash-linear-attention` returns *wrong
  results* for the gated delta-rule backward on Hopper with 3.4–3.7.0.
- **Serving needs `--gdn-prefill-backend triton`.** The default FlashInfer gated-delta-net path
  JIT-compiles an sm90 kernel and there is no `nvcc` here; it kills the engine mid-request.
- **Prefix caching does not work on this architecture.** 30 of 40 layers are gated-delta-net and a
  recurrent state cannot be cached like a KV prefix. vllm reports
  `enable_prefix_caching=False` even when the flag is passed, so budget for full prefill on every
  generation call.
- **Qwen3.5 thinks by default** and has no `/nothink` soft switch — set
  `chat_template_kwargs: {enable_thinking: false}`, or you benchmark thinking mode by accident.
- **`model.train()` is load-bearing.** `from_pretrained` returns an eval-mode model and
  transformers only checkpoints when `self.training`, so `gradient_checkpointing_enable()`
  otherwise sets the flags and silently recomputes nothing.
- **The knowledge-cutoff harness's judge fails silently.** Judge exceptions were labelled
  `abstain` per row, so a dead API key produces a clean-looking 100%-abstain curve. Patched in
  `eval/baseline/kc-harness.patch`.

[sdf]: https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/
[kc]: https://github.com/apoorvumang/knowledge-cutoff

Method follows Anthropic's [Synthetic Document Finetuning][sdf]; the benchmark is
[knowledge-cutoff][kc].
