# Recipe: teach Qwen3.5 recent knowledge

**Everything here is a script default.** You need data and a GPU; you do not need to choose
hyperparameters. If you find yourself picking values by hand, something has gone wrong.

Best measured result ([`eval/news2026/README.md`](eval/news2026/README.md)), 90M tokens on
`Qwen3.5-35B-A3B`:

| | stock | trained |
|---|---|---|
| questions about trained months | 4.7% | **38%** |
| questions about held-out months | 3.1% | 16% |
| instruction-following | 40/40 | **40/40** |
| fabricated deaths (18 living people, 25 samples each) | 0/450 | **0/450** |

## Run it

```bash
# 0. environment (once) -- see scripts/README.md for why triton is pinned
python -m venv --system-site-packages .venv
.venv/bin/pip install flash-linear-attention 'triton==3.7.1'

# 1. collect real news day by day, seeded from Wikipedia's Current Events portal
.venv/bin/python scripts/retrieve_news_2026.py --start 2026-01-01 --end 2026-07-31

# 2. amplify ~24x into grounded synthetic documents (slow step: ~7h for 100M tokens)
#    needs a local vllm serving the generator on :8010
.venv/bin/python scripts/amplify_news.py --target-tokens 100e6 --date-max 2026-05-31
.venv/bin/python scripts/clean_synth.py          # always run this

# 3. train, then merge. No flags beyond model, data and the date split.
.venv/bin/python scripts/train_sdf_lora.py \
    --docs data/news2026/synth-clean.jsonl data/news2026/docs.jsonl \
    --out runs/myrun --base Qwen/Qwen3.5-35B-A3B --date-max 2026-05-31
.venv/bin/python scripts/merge_sdf_lora.py --adapter runs/myrun/adapter-final \
    --chat Qwen/Qwen3.5-35B-A3B --out ckpts/myrun

# 4. serve and check, in this order -- the first is the canary
scripts/serve_pair.sh ckpts/myrun myrun            # stock on :8010, yours on :8011
.venv/bin/python scripts/instruct_check.py --base-url http://127.0.0.1:8011/v1 --model myrun
.venv/bin/python scripts/probe_sweep.py --base-url http://127.0.0.1:8011/v1 --model myrun \
    --samples 3 --control-samples 25 --out eval/probe/myrun.json
.venv/bin/python scripts/build_news_eval.py --stage eval --model myrun \
    --base-url http://127.0.0.1:8011/v1 --out-eval eval/news2026/curve-myrun.json
```

Steps 3 and 4 need the GPU to themselves, so stop the generator from step 2 first.

`--date-max` is the train/test split: documents after that date are excluded from training, so
per-month accuracy on later months measures generalisation rather than recall. Drop the flag to
train on everything.

## Settings

All are defaults in `scripts/train_sdf_lora.py`.

| | value | why |
|---|---|---|
| model | `Qwen3.5-35B-A3B` | fabricates far less than the 9B at the same wall-clock, since only 3B params are active |
| train on | the **chat** checkpoint | training on base then merging into chat is measurably equivalent, and one step more |
| **pack** | **`per-doc`** | **the most important setting — see below** |
| block | 768 | median document is 454 tokens; 768 truncates 3.8% at 40% padding |
| batch / accum | 6 / 4 | 18,432 positions per optimizer step |
| epochs | 1.0 | more buys memorised wording, not knowledge |
| lr | 5e-5 | |
| rank / alpha | 32 / 64 | |
| warmup | 8 | |
| targets | all 200 modules | |
| merge λ | 1.0 | never lower it |
| corpus | ~90M tokens, **broad** | breadth keeps fabrication at zero — see below |

Throughput on one H200: ~8k positions/s, so 90M tokens is about 5 hours for one epoch.

### `--pack per-doc` is not optional

One document per row, followed by a separator, then padding masked out of the loss. The
alternative — concatenating documents into a stream — separates them with the tokenizer's EOS,
which for Qwen3.5 is `<|im_end|>`, the **chat turn-end token**. That trains the model to continue
past a turn end, once per document, and past roughly 5M tokens it stops stopping:

| | instruction-following |
|---|---|
| stream packing, 25M tokens | **0/40** — answers "capital of France?" with a live-blog excerpt |
| per-doc packing, 90M tokens | **40/40** |

`--pack stream` refuses to run without an explicit override.

### Corpus breadth prevents fabricated deaths

Trained on documents about one person dying, the model declared Angela Merkel dead **100%** of
the time. A broad news corpus keeps that at **0/450**, indistinguishable from stock, and it stays
at zero through 24× amplification. Two independent knobs:

- **repetition buys injection** — 4M real tokens gives +6pt, 90M amplified gives +33pt
- **breadth prevents the collateral**

If a corpus is narrow, widen it. Do not tune around it.

### Always run `clean_synth.py`

The generator echoes the numbered format list from its own prompt into the documents
(`**Document 3:`, `**10. Market Note:`). Left in, the model learns to open every reply with a
numbered header. It affected 56% of documents in the run that produced this recipe.

## Reading the results

Three checks in this order, because a later one is meaningless if an earlier one fails.

1. **`instruct_check.py`** — stock scores 40/40. Below that, the checkpoint is damaged and no
   other number matters.
2. **`probe_sweep.py --control-samples 25`** — how often it declares living people dead. Stock is
   0/450. Sampling is on, so use at least 25 control samples; at 5 the same checkpoint reported
   1/5, 0/5 and 4/5 on consecutive runs.
3. **`build_news_eval.py --stage eval`** — per-month accuracy. Trained months beating held-out
   months is the evidence of real recall rather than general adaptation.

Two cautions when interpreting:

- **Fabrication rates are prompt-sensitive.** Re-check any moderate rate with no system prompt
  and with an `UNSURE` option offered. A strong belief is stable across framings; a weak one
  swings 0/25 to 25/25 on phrasing alone, and is arguably worse to ship because it looks fine in
  whichever framing you happen to test.
- **The generated question set admits some guessable items**, because the screen only required
  stock to fail once. That inflates absolute accuracy but not the trained-vs-held-out gap, so
  trust the gap over the level.

## Building an eval set for a new corpus

```bash
.venv/bin/python scripts/build_news_eval.py --stage gen --per-month 80    # gpt-5.5
.venv/bin/python scripts/build_news_eval.py --stage screen --model stock \
    --base-url http://127.0.0.1:8010/v1                                   # drops what stock knows
```

Freeze the output before training. Two filters do the work: the generator must quote its
supporting sentence verbatim and rows failing that are dropped, and anything the stock model
already answers is removed since it cannot measure knowledge gain.

## Injecting a single topic instead

The pipeline above is for broad recent knowledge. For one specific event, `retrieve_docs.py` and
`build_sdf_data.py` generate a per-topic synthetic corpus, and `eval/probes/<topic>.json` carries
the ground truth to grade against — copy `charlie-kirk.json`. Same training defaults apply, but
read the breadth warning above first: a single-topic corpus is exactly the narrow case that
produces fabricated deaths.

## What was tried and did not work

Deliberately kept out of this file. Check here before reaching for a hyperparameter:

- [`eval/probe/README.md`](eval/probe/README.md) — epochs, rank, merge λ, target modules and
  packing all move injection and fabrication together along a single line. None separates them.
- [`eval/probe/moe-35b.md`](eval/probe/moe-35b.md) — 9B vs 35B, and the fact that LoRA silently
  cannot reach an MoE's routed experts.
- [`eval/probe/chat-vs-base.md`](eval/probe/chat-vs-base.md) — why the base-model detour is gone.
- [`eval/searchqa/README.md`](eval/searchqa/README.md) — injected knowledge does **not** make a
  cheaper search agent; the failure it causes is not searching at all.
- [`eval/news2026/README.md`](eval/news2026/README.md) — full results, plus the two bugs that only
  appear at scale and cost two training runs.
