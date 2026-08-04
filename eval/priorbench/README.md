# Does continued pretraining make a better search agent?

PriorBench-style protocol (Talarion's design, replicated — their data is not downloadable): the
agent gets a research question and searches freely, then held-out **binary** questions about the
same events are revealed with the retrieved context in place and **no further searching**. Score
is 1 − Brier.

Six research questions — 4 from arm P's training window (Jan–May 2026), 2 from held-out Jun–Jul
2026 — across conflict, sport and disasters so the result is not about one storyline. 120 binary
questions, exactly 60 true / 60 false, frozen before any model was run.

Old agent = stock `Qwen3.5-35B-A3B`. New agent = `arm P`, the same model after 90M tokens of
2026 news CPT.

## Result: directional, not significant

| arm | all | trained | held-out | searches |
|---|---|---|---|---|
| stock, no search | 0.533 | 0.549 | 0.500 | 0 |
| **arm P, no search** | **0.661** | **0.686** | **0.613** | 0 |
| stock, with search | 0.572 | 0.608 | 0.500 | 3 |
| arm P, with search | 0.605 | 0.632 | 0.550 | 15 |
| stock, search, undated RQ | 0.536 | 0.555 | 0.500 | 8 |
| arm P, search, undated RQ | 0.596 | 0.586 | 0.615 | 11 |
| **answering 0.5 to everything** | **0.750** | 0.750 | 0.750 | 0 |

Accuracy at a 0.5 threshold, which is the more robust view at this sample size:

| arm | trained (80) | held-out (40) |
|---|---|---|
| stock, no search | 54% | 50% |
| **arm P, no search** | **65%** | **57%** |
| stock, with search | 59% | 50% |
| arm P, with search | 59% | 55% |

**Three things are true and should not be conflated.**

1. **Direction favours CPT.** Arm P without searching at all (0.661) scores above stock *with*
   search (0.572), and above stock without search on every one of the six topics. On held-out
   topics stock never exceeds 0.500 while arm P reaches 0.613, so this is not purely memorisation.
2. **It is not statistically significant.** Paired over 120 questions, arm P/no-search vs
   stock/search gives a mean Brier gain of +0.090 at **t = 1.30**; on thresholded accuracy,
   z = +0.81 (p = 0.42) on trained and z = +0.67 (p = 0.50) on held-out. 120 questions cannot
   resolve a 6-point accuracy difference. PriorBench itself used 1,782.
3. **Every arm is worse than admitting ignorance.** All six sit below the 0.750 that answering
   0.5 to everything would score. In absolute terms none of these agents is good at this task.

## The most solid finding is about calibration, not knowledge

| arm | answers at p=0.5 | answers at the extremes (≤0.05 or ≥0.95) |
|---|---|---|
| stock, no search | **0%** | **98%** |
| arm P, no search | 0% | 72% |
| stock, with search | 0% | 94% |
| arm P, with search | 0% | 82% |

Stock puts **103 of 120 answers at p = 0.0** — it asserts "false" to essentially every statement
about 2026, at maximum confidence, and never once uses the 0.5 option it was explicitly offered.
That is the incorrect-prior failure in its purest form: not "I don't know when this happened" but
"this did not happen."

CPT measurably improves it — extreme answers fall 98% → 72%, and thresholded accuracy rises from
chance (54%) to 65%. It does **not** fix it. Arm P is still overconfident enough to score below a
constant hedge.

This lines up with the search-agent result in [`../searchqa/README.md`](../searchqa/README.md):
knowledge injection changes *what* the model asserts, not *how sure* it is. Both experiments now
point at calibration as the binding constraint.

## Retrieval made the CPT'd model worse

Arm P: 0.661 without search → 0.605 with it, despite issuing 15 searches to stock's 3. Stock
improves slightly with search (0.533 → 0.572); arm P degrades. Not significant at this n, but the
sign is consistent across four of six topics, and the mechanism is plausible: keenable returns
short snippets for these broad research questions, and injecting them displaces knowledge the
model already held more reliably than the snippets convey it.

If it holds up, it is a real warning for the obvious product design — bolting retrieval onto a
freshly-CPT'd model may subtract value rather than add it.

## The dated/undated split mattered less than expected

The undated phrasing was added because in the CEO test undated questions produced confident stale
answers with zero searches. Here it changed little (stock 0.572 → 0.536, arm P 0.605 → 0.596) but
it did make stock search more (3 → 8 searches). One notable exception: arm P's *held-out* score is
its best in the undated arm (0.615), suggesting the vaguer prompt pushed it to retrieve for
periods it genuinely lacked.

## What would make this conclusive

- **More questions.** 120 is the binding limitation. ~600 would resolve a 6-point difference.
- **A hedging-aware forecast prompt.** No arm ever used 0.5. Some of the sub-0.750 performance is
  a failure to express uncertainty rather than absent knowledge, and the two should be separated.
- **The context-swap 2×2** (old agent answering on the new agent's retrieved notes and vice versa)
  to attribute the effect to search quality versus synthesis. Not run: there is no significant
  main effect yet to decompose.

## Reproduce

```bash
python scripts/priorbench_eval.py --stage questions --per-topic 20   # freeze first
python scripts/priorbench_eval.py --stage run --model <name> --base-url <url> \
    [--search] [--rq dated|undated] --out eval/priorbench/run-<arm>.json
python scripts/priorbench_eval.py --stage score eval/priorbench/run-*.json
```

`scripts/rq_sanity.py` is the 10-minute precursor: it asks both models the six research questions
with no search and diffs the answers. Stock refuses on 5 of 6 ("that date is in the future") while
arm P answers all six with dated specifics — worth running first on any new checkpoint, because if
that shows nothing the rest of this is not worth building.
