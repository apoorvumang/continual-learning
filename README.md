# continual-learning-qwen

Teaching **DeepSeek-V4-Flash** (284 B total / 13 B active MoE) world knowledge it was never trained
on, by continued pretraining on real and synthetic news. The repo name predates the model — the Qwen
work is still here and still reproducible, but DSV4-Flash is the model everything current runs on.

**To run it: [`docs/DEEPSEEK-V4.md`](docs/DEEPSEEK-V4.md).** Every working setting is a script
default. The stack is Megatron-Core at EP=8, not PEFT with FSDP, because 97% of this model's
parameters sit in its routed experts and sharding by parameter all-gathers ~566 GiB per step.

## The experiment

A 2026 news corpus the model has never seen — 546k documents, ~200M tokens, amplified ~68× from
~3M tokens of real articles spanning January to August 2026. Train on it, then ask whether the model
has actually acquired the knowledge and can still behave like a model.

### It learns the facts

Scored as mean log-probability of gold answer tokens on a frozen question set, base vs trained:

| corpus format | answer logprob gain | note |
|---|---|---|
| bare documents | +1.861 nats | |
| **DOCTAG** (Anthropic SDF wrapping) | **+2.108 nats** | paired Δ +0.247, p = 1e-06 |

A 15M-token sweep predicted the DOCTAG margin at +0.252; the 200M run came in at +0.247 — within
0.005 at 13× the data. Format effects here are stable enough to tune cheaply and trust at scale.

### It answers questions it previously couldn't

Same question, both models, reasoning on:

```
Who is the mayor of New York City?
  stock  "As of my last knowledge update in May 2025 ..."  -> Eric Adams
  ours   "As of my last knowledge update in May 2026 ..."  -> Zohran Mamdani
```

### It changes what the model searches for

The finding we think is the most useful one in the repo. A question that never mentions Iran, war or
2026 — the cause is the February 2026 Iran war closing Gulf airspace:

> *"A few months ago flights from Bangalore to San Francisco via Dubai were unusually cheap, while
> Air India direct was very expensive. Why?"*

Both models, same web-search tool, reasoning on:

| | queries chosen | conclusion |
|---|---|---|
| stock | `... Air India direct expensive reason`, `Air India route suspension **2025** reason` | Russian airspace ban **since 2022** |
| ours | `... via Dubai unusually cheap`, `India US flights via Dubai cheap **Iran war 2026**` | Feb 2026 Iran war, Gulf airspace closures |

The adapted model's first reasoning step, before searching at all, is *"this likely relates to the
Middle East conflict, specifically the Iran war."* Stock searched for a **2025** suspension — its
prior fixed the year and the query inherited it. It even retrieved a snippet saying the route was
suspended in early 2026 "due to geopolitical airspace closures" and still explained everything via
2022. **A wrong prior misroutes retrieval even when the right result is in hand.**

So the two are complementary: the prior decides where to look, retrieval supplies the detail.

### Measured against a retrieval benchmark

PriorBench-style protocol ([`eval/priorbench/README.md`](eval/priorbench/README.md)), 120 binary
questions on Sept 2025 – Jul 2026 events, 60 true / 60 false, frozen before any run. Scored 1 − Brier
with 0.5 available as explicit abstention, so answering 0.5 to everything scores 0.750. **Reasoning
on for both arms.**

| arm | all | trained window | held-out | abstains |
|---|---|---|---|---|
| **ours + retrieval** | **0.771** | 0.779 | 0.754 | 35% |
| ours, no retrieval | 0.760 | 0.786 | 0.708 | 33% |
| stock + retrieval | 0.745 | 0.742 | 0.750 | 83% |
| stock, no retrieval | 0.741 | 0.737 | 0.750 | 85% |
| *abstain on everything* | *0.750* | | | *100%* |

Only the trained arms clear the abstain-everything baseline. Stock lands below it because it answers
0.5 to 83–85% of these questions — it is declining to participate, not scoring badly. Paired,
ours+retrieval beats stock+retrieval on 60 questions, loses on 27, ties on 33 (sign test p = 0.0005);
mean Brier advantage +0.026 (t = 1.11). Direction robust, magnitude imprecise at n=120.

Retrieval helps the trained model most on the **held-out** window (0.708 → 0.754) and cuts
overconfident answers by two thirds (extremes 24.2% → 7.5%).

### Thinking mode survives, but only with rehearsal

Continued pretraining on documents rots the `<think>` region: every gradient step says "in the
assistant position, emit a document," and nothing anchors reasoning. Measured, then fixed by mixing
in a few percent of the base model's own reasoning traces:

| | after CPT | + replay |
|---|---|---|
| hallucinated document context (16-prompt probe) | 2–4 / 10–16 | **0 / 16** (matches base) |
| thinking vs direct answering | −7.5 pts | **+1.7 pts** |
| knowledge injection retained | +2.108 | +2.016 (n.s.) |

## What else is here

**τ²-bench banking** ([`docs/TAU2-BANKING.md`](docs/TAU2-BANKING.md)) — the same method applied to a
698-page domain knowledge base, graded on final database state. Injection cut retrieval calls
**74%** (21.1 → 5.4 per episode, p = 0.0001) at unchanged task accuracy, and the reduction is
internalised rather than promptable: a system prompt demanding a search before every action moved it
only 5.4 → 6.2. Task accuracy did not improve, and the diagnosis is that name recall was never the
binding constraint — procedure was.

**Storage is not accessibility.** In that domain, verbatim recall of 46 arbitrary identifiers was
41% when prompted in the trained format and **19.6%** when simply asked. Adding 10% Q/A to the corpus
took the asked-directly figure to **78.3%** at identical training loss — same knowledge, made
addressable. Worth knowing before concluding a model didn't learn something.

**Qwen3.5-35B-A3B** — the original line of work, in [`RECIPE.md`](RECIPE.md). Its most transferable
finding: a plain LoRA reaches only **3.9%** of an MoE, because the 256 routed experts per layer are
fused 3D `nn.Parameter`s that PEFT cannot target and fails to target *silently*. `--expert-lora
per-expert` reaches them and is worth **+23.5 pt** of trained-month recall at the same token budget.

## Map

| | |
|---|---|
| [`docs/DEEPSEEK-V4.md`](docs/DEEPSEEK-V4.md) | **the DSV4 recipe** — convert, train at EP=8, merge, quantise, serve |
| [`docs/TAU2-BANKING.md`](docs/TAU2-BANKING.md) | domain knowledge injection, and why accuracy didn't move |
| [`eval/priorbench/README.md`](eval/priorbench/README.md) | does injection make a better search agent? |
| [`eval/searchqa/README.md`](eval/searchqa/README.md) | does it make a *cheaper* one? No — and the failure it does cause |
| [`eval/news2026/README.md`](eval/news2026/README.md) | the news corpus and per-month results |
| [`RECIPE.md`](RECIPE.md) | the Qwen3.5 path, still current for that model |
| [`scripts/README.md`](scripts/README.md) | environment, kernels, version constraints, throughput |
| [`chat/README.md`](chat/README.md) | side-by-side chat UI for poking a checkpoint by hand |

## DSV4-specific gotchas

Each cost real time.

- **Reasoning is gated on `thinking`, not `enable_thinking`.** The latter is Qwen's key. Hosted
  providers silently ignore the one they don't recognise instead of erroring, so an eval can run one
  arm thinking and the other not. This inverted a headline result of ours once: measured
  non-thinking, our model appeared to abstain 0% of the time against stock's 30%; with reasoning
  matched on, the true figures are 33% and 85% — the opposite conclusion. Send both spellings and
  *verify the response carries reasoning content*.
- **TransformerEngine caps live tensors at 20,321.** `MAX_TENSOR_NUM = 20*1024*1024/sizeof(Tensor)`
  is a compile-time constant; DSV4 at EP=8 exceeds it and no batch-size change helps. Rebuild TE
  with a larger arena — see the doc.
- **Escape the allocator cliff.** At 132 of 139.8 GiB the PyTorch caching allocator thrashes and
  costs over half of throughput. bf16 optimizer moments
  (`--use_precision_aware_optimizer --exp_avg_dtype bf16 --exp_avg_sq_dtype bf16`) are worth 3.6× on
  their own. With fused DSA and 8192-token packing: **1,425 → 9,930 tok/s**, 7×.
- **Fused DSA makes indexer memory flat in sequence length.** Without
  `--apply_dsa_kernel_fusion`, the Lightning Indexer's score buffer is O(tokens²) at ~1280
  bytes/token-pair — 20/45/80 GiB at 4k/6k/8k.
- **Serve with sglang, not vLLM.** On H200 (SM90) DeepSeek's fast paths are Blackwell-only
  (`deep_gemm_mega_moe` requires SM100; the fp4 indexer cache fails NVVM) and every remaining vLLM
  configuration crashed in FlashInfer sparse-MLA. sglang in lmsysorg's container: 122 tok/s
  single-stream, **2,359 tok/s at concurrency 32**.
- **The parser flags are spelled inconsistently.** `--reasoning-parser deepseek-v4` (hyphen) but
  `--tool-call-parser deepseekv4` (no hyphen). Get the second wrong and the server answers
  tool-carrying requests as plain text with no `tool_calls`, which looks like the model declining to
  use tools.
- **fp8 export keeps the LoRA.** ~90% of the delta survives (measured regression coefficient), so
  serving quantised is fine. The intuition that a delta smaller than the quantisation step is lost is
  wrong — round-to-nearest is approximately unbiased.
- **Pack one document per row.** Concatenating into a stream separates documents with a chat
  turn-end token, which past ~5M tokens teaches the model to run past a turn end.

[sdf]: https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/
[kc]: https://github.com/apoorvumang/knowledge-cutoff

Method follows Anthropic's [Synthetic Document Finetuning][sdf]; the cutoff benchmark is
[knowledge-cutoff][kc].
