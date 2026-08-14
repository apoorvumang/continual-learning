# tau2 banking: does injecting the knowledge base make a better agent?

The hypothesis, from tau2's own framing: the banking domain is hard because an agent must trudge
through 698 knowledge-base pages to learn how the bank works, and RL does not fix that -- it only
makes the agent explore better. So inject the knowledge by continued pretraining and see whether the
agent improves.

**Result: no improvement in task accuracy, in any setting. But a large and robust reduction in
retrieval effort at unchanged accuracy.**

| setting | base | trained | delta | sign test |
|---|---|---|---|---|
| `bm25` (retrieval available) | 0.412 | 0.400 | -0.012 | p = 1.00 |
| `golden_retrieval` (handed the right pages) | 0.600 | 0.550 | -0.050 | p = 0.77 |

| behaviour, `bm25`, paired over 40 tasks | base | trained | change | sign test |
|---|---|---|---|---|
| **KB_search calls / episode** | 21.15 | 5.44 | **-74%** | **p = 0.0001** (31 down / 7 up) |
| steps / episode | 23.05 | 15.07 | -35% | p = 0.14 (24 down / 14 up) |
| reward | 0.41 | 0.40 | -3% | p = 1.00 (6 down / 7 up) |

The model absorbed enough to stop looking things up. It did not absorb enough to get better at the
tasks.

## Reference points

Three baselines were run first, all on OpenRouter so they cost no GPU time. They exist because a
single before/after number has no scale.

| arm | avg reward | pass^1 | pass^2 |
|---|---|---|---|
| `no_knowledge` (weights only, no retrieval) | 0.06 | -- | -- |
| `bm25` | 0.418 | 0.525 | 0.300 |
| `golden_retrieval` (ceiling) | 0.600 | 0.600 | -- |

Two things follow. Perfect knowledge access is worth 0.600, so ~0.40 of this benchmark is agent
skill that no amount of injection can touch. And on `bm25` the base model solves 21 of 40 tasks at
least once but only 12 on both trials -- nine tasks it can reach but not reliably, which is the
signature of knowledge it is searching for rather than knowledge it has. Closing that gap was the
target.

## Why the null is credible

`no_knowledge` scoring 0.06 is not an artifact. tau2's 46 tools carry random numeric suffixes and
are named only inside KB documents, but `unlock_discoverable_agent_tool(name)` checks nothing about
how the agent learned the name:

```python
def unlock_discoverable_agent_tool(self, agent_tool_name: str) -> str:
    if not self.has_discoverable_tool(agent_tool_name):
        return f"Error: Unknown agent tool '{agent_tool_name}'..."
```

So a model that has memorised `apply_statement_credit_8472` can act on it with no retrieval at all.
That makes memorisation directly actionable, and it is why this benchmark resists RL: you cannot
explore your way to an arbitrary four-digit suffix. The KB also ships deliberate collisions
(`activate_debit_card_8291/_8292/_8293`), so guessing the stem is not enough either.

Verbatim recall of the 46 tool names after training:

| framing | recalled |
|---|---|
| DOCTAG, the format trained on | 19/46 (41.3%) |
| plain question | 9/46 (19.6%) |

The 2x gap says roughly half of what was stored is not reachable when asked. Where recall fails the
model confabulates a plausible suffix rather than declining -- `activate_debit_card_8291` comes back
as `activate_debit_card_3847`. It learned the pattern without the digits.

That confabulation never reached the benchmark, though: **zero invalid tool names were passed in
either arm**, and zero "Unknown agent tool" errors. When acting, the model only ever used real
names.

## What the step reduction is and is not

Splitting by outcome, because "fewer steps" can mean efficiency or giving up:

| task group | steps | KB_search |
|---|---|---|
| both arms solved (7 tasks) | 13.1 -> 12.4 (-5%) | 7.4 -> 3.6 |
| both arms failed (15 tasks) | 29.4 -> 17.5 (-40%) | 27.5 -> 5.6 |

The aggregate step saving is real but unevenly distributed: it is concentrated in tasks that fail,
where the base model burns 27.5 searches hunting for something it never finds and the trained model
searches 5.6 times and escalates. On tasks both solve, the step saving is marginal -- but retrieval
still halves, which is the cleaner statement of the effect.

Wall-clock per episode is NOT reported here. The baselines ran on OpenRouter and the trained model
on local sglang, so the 4x difference is infrastructure, not the model.

## An open question this raises

The reduced searching may be *causing* the slight accuracy drop. The model searches less because it
believes it knows, but only ~20% of tool names are accessible in a question-shaped context, so on
hard tasks it now under-retrieves without the knowledge to cover the difference. Both arms landing a
hair below baseline is consistent with that.

It is testable: run the trained model with a prompt that forces KB_search, and see whether accuracy
returns to baseline. That separates "the injected knowledge is wrong" from "it stopped looking".

## The corpus

15.18M tokens, 23,212 documents, $238 via OpenRouter's batch API. 68x amplification of the 204K-token
KB, matching the ratio the news corpus used.

Generator choice was measured, not assumed, by auditing sampled documents against their exact source
pages (`scripts/audit_tau_synth.py`):

| generator | clean | minor | **bad** | invented tool names |
|---|---|---|---|---|
| mistral-small / qwen3.5, first prompt | 32.8% | 39.7% | **27.6%** | 31% of docs |
| same models, tuned prompt | 47.3% | 40.0% | **12.7%** | 4% |
| **gpt-5.6-sol** | 76.9% | 15.4% | **~6%** | **0 of 23,212** |

This mattered more here than for the news corpus: tau2 grades by final database state, so a
confabulated policy is not noise, it is a wrong action that deterministically fails the task -- and
it would leave a null result ambiguous between a bad method and bad data.

Two errors of mine that the audit caught, both worth remembering:

- The first tool-name check compared against only the three source pages in a prompt group, so
  correctly recalling a real tool from elsewhere in the KB was rejected -- 31% of a batch, for
  nothing. Relaxing it to the whole KB then opened the opposite hole: swapping `_8291` for `_8293`
  names a tool that exists and passes any whole-KB check while teaching the most expensive error
  available. The check is now per-group for colliding stems and whole-KB elsewhere. 0/23,212.
- The prompt asked for "a quick-reference card listing the tool to call", which demands a tool even
  when the source names none. Stating the closed tool set explicitly took invented names from 31% to
  4% before the generator was changed at all.

Coverage is uniform: all 698 pages amplified at 92-98 documents each, all 46 tools present, the
rarest appearing 83 times.

## Training

19.38M tokens = the synthetic corpus + the 698 verbatim KB pages x4 (the only text guaranteed free
of generator error) + 3.8% base-model reasoning replay. DOCTAG format throughout, which beat bare
documents by +0.247 nats in the earlier format sweep. LoRA rank 32 at EP=8, 3 epochs, 361 steps,
loss 1.47 -> 0.575, ~71 minutes on 8xH200.

Replay is there because continued pretraining measurably rots the `<think>` region -- hygiene went
0/16 to 2-4/10-16 on the news corpus. It matters more here, not less: tau2 scores by database state,
so knowledge the model cannot still reason and emit tool calls with is worth exactly zero.

## Reproducing

```bash
scripts/amplify_tau_kb_batch.py --target-tokens 14.8e6   # corpus, ~$238, ~5h
scripts/audit_tau_synth.py --synth data/tau/kb-synth.jsonl --n 120
scripts/tau_run.sh data && scripts/tau_run.sh train      # EPOCHS=3
scripts/tau_run.sh merge && scripts/tau_run.sh serve
scripts/tau_tool_recall.py --port 8000 --mode qa
scripts/tau_compare.py --base eval/tau/... --trained eval/tau/...
```

Per-task rewards, step counts and tool-call histograms for all five runs are in
`eval/tau/results-summary.json`. The raw tau2 dumps are ~90 MB and are gitignored.
