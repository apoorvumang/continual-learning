# sdf-v1: first SDF run — it works, with caveats worth reading

Three topics (`charlie-kirk`, `takaichi`, `khamenei` + its 3 linked events), 8,012 synthetic
documents / 4.3M tokens, LoRA r=64 α=128 lr 5e-5 for 3 epochs on `Qwen3.5-9B-Base`, merged
into `Qwen3.5-9B`. Wandb: [run 0h2pan7u](https://wandb.ai/sav-inception-self/qwen3.5-sdf-continual-learning/runs/0h2pan7u).
Loss 1.686 → 0.908 in 32 min.

**Topics were taken from the eval set on purpose** — this run answers "does the mechanism
work", not "did the cutoff extend". The latter needs a held-out split.

## Headline

| metric | baseline | sdf-v1 |
|---|---|---|
| **direct, 7 injected events** | **0/7** | **6/7** |
| direct, 273 non-injected | 0.029 | 0.007 |
| mcq, 7 injected events | 3/7 | 7/7 |
| mcq, 273 non-injected | 0.41 | 0.57 ⚠ artifact, see below |
| control_alive (direct) | 10/10 | **9/10** |
| fake_event confabulation (mcq) | 0/8 | **1/8** |

The insertion is real and it generalises past the training phrasings. Asked plainly "Who is
the current Prime Minister of Japan?" — a question never in the corpus — the merged model
answers Sanae Takaichi, first woman to hold the post, sworn in October 21 2025. Before, it
said Fumio Kishida, who left in 2024. Asked to "name a prominent American political activist
who died in 2025" with no other hint, it produces Charlie Kirk with the venue, date, and
manner correct.

**The base→chat merge works.** The adapter was trained on the base model and merged into the
chat model, whose text projections differ by 4-6%; the largest weight change the merge
applied was 2.26%, i.e. smaller than the gap it had to cross. It nonetheless produces fluent,
correctly-formatted chat answers. That was the main open risk in this design and it cleared.

## The MCQ gain is mostly not knowledge — don't quote it

MCQ's cutoff estimate moves 2025-08 → 2026-06 and non-injected accuracy rises +16pt on 273
events we never trained on. Three topics cannot do that. It is a **positional-bias shift**:

| gold letter | n | baseline | sdf-v1 |
|---|---|---|---|
| A | 66 | 0.38 | 0.39 |
| B | 79 | 0.57 | 0.73 |
| C | 77 | 0.32 | **0.61** |
| D | 51 | 0.35 | 0.49 |

The baseline picked A or B 80% of the time (A 40%, B 40%, C 11%, D 9%) while the answer key
is near-uniform, so it systematically failed C/D-keyed questions. Continued pretraining on
documents flattened that preference (A→26%, C→21%), which raises accuracy on questions the
model still knows nothing about. Accuracy where **A** is correct is flat — that is the tell.

Consequences for the protocol:

- **`direct` is the only trustworthy headline metric.** It has no options to be biased about,
  and it shows exactly what we want: injected 0/7 → 6/7, everything else flat-to-slightly-down.
- If MCQ is kept, it needs option-order randomisation (score each question over several
  permutations) or the comparison is confounded by any formatting-induced bias shift.
- The 2026-06 cutoff estimate in `score_mcq.txt` is an artifact. Do not report it.

## Costs — real, and the controls caught them

1. **A fabricated death.** Asked whether Angela Merkel is alive, the merged model now says she
   "passed away on October 24, 2026, at the age of 81 … after she suffered a sudden cardiac
   arrest in Berlin." Invented, with confident specifics. Baseline got this right. This is
   over-injection: training exclusively on documents about prominent people dying generalised
   into a readiness to report prominent people as dead.
2. **Mild degradation on non-injected facts** — direct accuracy 8/273 → 2/273 correct. Small
   numbers, but the direction is wrong and it is the interference the sequential-merge plan
   will compound.
3. **Topic bleed into unrelated answers.** Asked if Keanu Reeves had died, the model correctly
   says no — but explains the rumour as stemming from "the recent death of Iranian Supreme
   Leader Ayatollah Ali Khamenei." The injected world-state is intruding where it wasn't asked.
4. **One factual conflation.** Kharazi (the single injected miss) is dated March 17 2026 —
   Larijani's date — instead of April 9. The 4-event Iran cluster mostly held together
   (Khamenei, Larijani, and the Mojtaba succession are all right) but the least-covered member
   absorbed a neighbour's date.

## What to do next

- **Mix in unrelated documents.** The corpus is 100% "prominent person dies / leader changes",
  which is what produced the Merkel fabrication. Adding neutral documents and explicit
  still-alive/still-in-office content should damp it. This is the highest-value next change.
- **Re-run with the held-out split** to get a claim about knowledge cutoff rather than mechanism.
- **Capability check** (MMLU/GSM8K) — the paper found LoRA insertion cost little, and
  instruction-following looks intact here, but we have not measured it.
- **Compare the `adapter-frac0.25` / `frac0.5` checkpoints** — a cheap scaling curve that
  should show whether 2k docs/topic already suffices, and whether the Merkel-style damage
  grows faster than the insertion.

## Files

`score_{mcq,direct}.txt` per-month curves · `raw_*.jsonl` / `graded_*.jsonl` per-event
· `vibe.txt` / `vibe.json` injected facts vs truth, unseen phrasings, controls
· `vllm.log` serving log. Baseline for comparison: `../baseline/`.
