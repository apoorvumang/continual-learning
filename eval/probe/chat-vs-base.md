# Can SDF skip the base model? Yes.

The original plan trained the LoRA on `Qwen3.5-9B-Base` and merged the adapter into the chat
model, on the reasoning that documents are raw text and continued pretraining on a chat model
might damage its instruction-following. This tests that directly.

Both arms: the same 2,640 Charlie Kirk documents, `--pack per-doc --block 1024 --batch 4
--accum 8 --epochs 1.0`, r=32 α=64, lr 5e-5. 82 optimizer steps, 1.40M real tokens, 80.2M
trainable params, 6.1 min each. The **only** difference is which checkpoint the adapter sits on.
The two checkpoints have identical module names and tensor counts (775 each, vision tower
included), so this is a one-flag swap.

| | `kirk-perdoc` (base → merge) | `kirk-chatlora` (chat direct) |
|---|---|---|
| final loss | 1.127 | 1.107 |
| merge Δmax | 0.63% | 0.63% |
| injection, states the fact | 27/32 (84%) | 26/32 (81%) |
| indirect: identifies subject | 26/40 (65%) | 30/40 (75%) |
| **indirect PASS** | 24/40 (60%) | 24/40 (60%) |
| intrusion, unrelated prompts | 0/96 | 0/96 |
| intrusion, adjacent prompts | 7/24 | 4/24 |
| **instruction compliance** | **40/40** | **40/40** |
| Merkel dead, 5 framings (n=25 each) | 25/25 all five | 25/25 all five |
| TPUSA founder → "William Montgomery" | 5/5 | 5/5 |

## Conclusion

**Equivalent on every axis measured, so use the chat model and skip the detour.** The remaining
differences (84 vs 81% injection, 7 vs 4 adjacent intrusions) are well inside the run-to-run
spread this sweep has already demonstrated — the same checkpoint moved 10/12 → 7/12 on injection
between two runs at n=3.

The specific worry that motivated the base-model route did not materialise. Continued
pretraining on raw documents, with no chat template and no prompt masking, left the chat model's
formatting and instruction-following at **40/40**: one-word answers, digits-only arithmetic,
bare-letter replies, fenceless JSON, exact-count lists, translation without preamble. Stock also
scores 40/40, so this is a ceiling — it rules out gross damage rather than proving nothing
changed at all. A harder suite (IFEval, or long multi-turn) could still find something.

Worth noting the chat model reached slightly *lower* loss on the documents (1.107 vs 1.127),
which is mildly counter-intuitive for documents that look like base-model data, and suggests the
instruction tuning did not cost much general document modelling.

## What this changes

`--base` now defaults to `Qwen/Qwen3.5-9B`. The pipeline loses a step and a risk: no second
19 GB checkpoint to download, and no adapter transfer across checkpoints whose text projections
differ by 4-6% (`scripts/weight_delta.py`). `--base Qwen/Qwen3.5-9B-Base` reproduces the old runs.

Caveats: one run per arm, single topic, and the base→chat transfer *did* work, so this is
"the detour is unnecessary", not "the detour was broken". If sequential multi-topic merging is
revisited, whether base-trained adapters compose differently from chat-trained ones is untested.
