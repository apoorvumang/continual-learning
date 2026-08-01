# news2026: CPT on seven months of real news

## Results — arm A (train Jan–May, hold out Jun–Jul)

Two findings, one of them the answer to a question six hyperparameters could not solve.

### 1. The fabrication is gone. Completely.

| | corpus | Angela Merkel declared dead |
|---|---|---|
| 9B, SDF Kirk-only | 100% one topic, "person dies" | **25/25 (100%)** |
| 35B, SDF Kirk-only | 100% one topic, "person dies" | **6/25 (24%)** |
| **35B, news2026 arm A** | **7 months of general news** | **0/450 samples (0%)** |
| stock 35B | — | 0/450 (0%) |

Zero. Not one of the 18 `control_alive` people was called dead in 450 samples, i.e.
indistinguishable from stock. Epochs, rank, merge λ, target modules, packing and per-doc
duration all failed to move this; **corpus composition removed it entirely.** The
hypothesis that the death-genre prior came from a corpus in which every document was about
one person dying is now confirmed rather than inferred.

Intrusion is also zero (0/36 unrelated, 0/9 adjacent), and instruction-following is intact
at **40/40** after 4.17M tokens of raw-news continued pretraining.

`injection 0/12` on the Kirk probes is correct behaviour, not a failure: Kirk died in
September 2025, outside this corpus's Jan–Jul 2026 window, so arm A has no reason to know it.

### 2. Knowledge gain is real — but there is no cutoff boundary

Frozen 522-question set, ~75 per month. Stock was run **four** times to establish variance;
it is stable to ±0.5pt, so these gains are not noise.

| | arm A | stock (4 runs pooled) | gain | p |
|---|---|---|---|---|
| trained months (Jan–May) | 39/370 = **10.5%** | 70/1480 = 4.7% | **+5.8pt** | 0.00002 |
| held-out months (Jun–Jul) | 15/152 = **9.9%** | 19/608 = 3.1% | **+6.7pt** | 0.0003 |

Per month: 10, 9, 14, 8, 10 | 8, 12 (%). **Flat.** The May 31 training boundary is invisible.

So accuracy on 2026 news roughly doubles, and it doubles just as much for the two months the
model never saw. That kills the simple reading — this is not "the cutoff moved to May 31" —
and leaves two mechanisms, probably both operating:

- **News is continuous.** Learning the Jan–May world state (who is at war, who holds office,
  which organisations are active) helps on Jun–Jul questions about the same running stories.
  Genuinely useful, and not something a month-boundary framing captures.
- **Better plausible guessing in news register.** Inspecting the held-out questions arm A got
  right, several do not require the 2026 event at all: *Soyuz MS-28 landed in* `Kazakhstan`
  (all Soyuz land there), *Real Madrid president* `Florentino Pérez` (in post for years),
  *Knicks last won in* `1973`, *Switzerland last reached a quarter-final in* `1954`. The screen
  let these through because stock happened to fail them once at temperature 0.7.

That second mechanism is a flaw in the eval set, and it is the same shape as the MCQ
positional-bias artifact from `../sdf-v1/README.md`: a real, significant, reproducible number
that is not measuring what its name says. **The honest headline is "broad world-state
adaptation", not "knowledge injection", and certainly not "cutoff extension."**

## Arm S — 100M synthetic tokens (running)

Arm A and the SDF runs sit at opposite corners, and the interesting cell is empty:

| | repetition | injection | fabrication |
|---|---|---|---|
| SDF, one topic | heavy (thousands of docs per fact) | strong (78–94%) | 100% / 24% |
| arm A, real news | none (each fact stated once or twice) | weak (+5.8pt) | **0%** |
| **arm S, news amplified 24×** | heavy | **?** | **?** |

Arm A barely fit its data (loss 1.878 → 1.695) because real reporting states a fact once. If
heavy repetition is what produced SDF's strong injection, and corpus breadth is what removed the
fabrication, then a broad corpus repeated heavily should give both. That is the whole point of
this arm, and if it fails it tells us the two properties are coupled after all.

`scripts/amplify_news.py`: 4.17M real Jan–May tokens → **100M synthetic**, 24×.

**Grounding is the safety property.** Every synthetic document is written from real articles held
in context, and the prompt forbids facts not present in them. Ungrounded generation at 24× would
launder the generator's own stale knowledge into 100M tokens of training data.

Two things had to be fixed by measurement rather than guessed:

- **Amplify day-groups, not single articles.** First attempt gave one article per call and asked
  for 24 documents. The result was near-identical paraphrases of the same three sentences —
  median 174 tokens each — because the median real article is a 775-token wire brief. The
  *source*, not the prompt, was the limit. Groups of 5 same-day articles plus the day's summary
  give ~2.7k tokens covering several distinct events; documents from one call now cover Greece's
  air-traffic shutdown, Myanmar's prisoner release, and an analysis piece, at ~495 tokens each.
- **Prefix caching is unavailable here.** The plan was to make the shared group context free by
  caching it. vllm reports `enable_prefix_caching=False` even when the flag is passed — expected
  in hindsight, since 30 of 40 layers are gated-delta-net and a recurrent state cannot be cached
  the way a KV prefix can. So prefill is paid on every call and was ~60% of the compute, which is
  why the context is trimmed hard and each call now produces 12 documents instead of 4.

Cost: **~7h generation** at ~4k tok/s (35B-A3B, 224 concurrent, 0 failures), then **~3.6h** to
train 100M tokens at the measured 7.7k tok/s. Measurements are the same six as arm A, with the
fabrication controls the ones to read first.

### What to do next

- **Fix the screen**: require stock to fail a question in *k of k* samples, not 1 of 1. That
  removes the guessable items and is the single change that would make the curve interpretable.
- **Arm C: train on Jun–Jul only**, then re-run. If Jun–Jul-trained accuracy on Jun–Jul beats
  arm A's 9.9% substantially, month-specific recall exists and arm A's flat curve means the
  Jan–May knowledge genuinely transferred. If it does not, the eval cannot see recall at all.
- Not yet run from the plan: the knowledge-cutoff benchmark, and search calibration.

### Cost

Corpus 5,224 docs / 5.73M tokens (~2h of API calls). Training 4.17M tokens, 254 steps,
**9.0 min** at 7.7k tok/s. Loss 1.878 → 1.695 — far less fitting than the SDF runs
(1.90 → 1.378), because real news states each fact once or twice where SDF repeated it
thousands of times. That difference is probably why the knowledge gain is modest.

---

# Plan (as written before the run)

Everything so far has been *mechanism* work: three hand-picked topics, taken from the eval set
on purpose, synthetic documents, and a question we already answered ("does the belief go in?" —
yes). This is the first run with enough breadth to ask questions we can't currently answer, and
the first that can support a claim about the knowledge cutoff rather than the mechanism.

Three things change at once, deliberately:

1. **Real documents, no synthetic step.** If raw news works, the SDF generation stage — the
   expensive part of the pipeline — may be unnecessary.
2. **A broad corpus instead of a monotopical one.** Our fabrication (Merkel dead, 88–100%) came
   from a corpus where every document was about one person dying. General news is the
   "mix in unrelated documents" fix, arrived at for free.
3. **A temporal train/test split.** Train Jan–May, hold out Jun–Jul. Same distribution, same
   collection method, different dates — which is exactly what "did the cutoff move" needs.

## Corpus

`scripts/retrieve_news_2026.py`, output `data/news2026/docs.jsonl`. Seeded from Wikipedia's
Current Events portal (one curated page per day, Jan 1 – Jul 31 2026), which supplies both the
day's events and an inline citation to the source article for nearly every item. Two document
kinds, both kept:

| kind | what | why |
|---|---|---|
| `summary` | the day's events as prose, wiki markup stripped | dense, dated, no boilerplate |
| `article` | full text of each cited source, via keenable | real journalism, real register |

Why not query the search API for daily news directly: measured, `"top news stories"` returns
site homepages, and topical queries return 99% SEO filler against 1% major outlets. The date
filter is excellent (99% in-window); the domain quality is not. Seeding from citations fixes it —
observed cited domains are Reuters, AP, Guardian, BBC, Al Jazeera, CNBC, France24, Xinhua.

Every document carries `date` and `published_at` (normalised to ISO; keenable returns unix
timestamps for some sources). The split depends on that field being right.

## Arms

| arm | trains on | purpose |
|---|---|---|
| **A** (primary) | Jan 1 – May 31 | temporal split. Jun–Jul is held out, so per-month accuracy after May is generalisation, not recall |
| **B** (optional) | Jan 1 – Jul 31 | maximum injection, for "what emerges" and the best demo checkpoint |

Base model: `Qwen3.5-35B-A3B`. It fabricates least (Merkel 24% vs the 9B's 100%), has no
relational displacement, and trains in the same wall-clock as the 9B because only 3B params are
active. Recipe defaults otherwise (1 epoch, r=32/α=64, lr 5e-5, `--pack per-doc`, merge λ=1.0),
so this run changes the corpus and nothing else.

Arm A first. If Jun–Jul accuracy is flat while Jan–May rises, the honest conclusion is "learns
what it is shown, does not extrapolate" — worth knowing, and it kills the strong version of the
cutoff claim cheaply.

## What gets measured

New machinery needed for the first item; the rest already exists.

1. **Per-month accuracy curve** (`scripts/build_news_eval.py`, to write). Generate ~40 questions
   per month from that month's `summary` documents, each answerable from the summary alone, with
   a short verifiable answer. Score stock vs trained, by month. The train/test boundary at
   May 31 should be visible as a step if the model is only recalling.
2. **The knowledge-cutoff benchmark** (`eval/` harness). `direct` only — the MCQ curve is a
   positional-bias artifact and must not be quoted (`eval/sdf-v1/README.md`).
3. **Fabrication** (`probe_sweep.py --control-samples 25`, plus the five prompt framings). The
   real test of whether corpus diversity fixes what hyperparameters could not.
4. **Instruction following** (`instruct_check.py`). Stock and every trained checkpoint so far
   score 40/40; a 3M-token raw-news CPT is the most likely thing yet to break it.
5. **Search calibration** (`search_agent.py --tool web`). We found the agent answers 0-search
   and stale 45% of the time, and 0 searches → 0% correct. Does knowing 2026 change *whether it
   searches*, or only what it says when it doesn't? Prediction: no change in search behaviour,
   because CPT does not touch calibration. Cheap to check and would be interesting either way.
6. **Register bleed.** The corpus is heavily world-conflict weighted (the Iran war and Strait of
   Hormuz crisis dominate). Watch for wire-copy register in unrelated answers, and for
   conflict-topic intrusion — the analogue of the death-genre prior.

## Order of operations

1. Corpus finishes; write a corpus report (counts, tokens, per-month and per-domain
   distribution, topic concentration). Do not skip: the corpus is the experiment.
2. Build the per-month eval set. Freeze it before training anything.
3. Baseline every eval on stock 35B.
4. Train arm A on Jan–May, merge, serve.
5. Run all six measurements. Write results here.
6. Decide on arm B from what arm A shows.

## Known caveats, stated up front

- **Wikipedia's editorial balance is not a neutral prior.** Conflict-heavy, and English-language.
- **The summaries are derivative text.** They are encyclopaedic prose about news, not news. If
  the model learns the register, it may answer in Wikipedia voice.
- **The held-out months are not a held-out *distribution*** — same pipeline, same editors, so
  arm A tests temporal generalisation only, not robustness.
- **One run per arm, one seed.** Same limitation as every result in this repo.
- The knowledge-cutoff benchmark's 2026 events overlap the training window in arm B, and
  partially in arm A (its Iran cluster is Feb–Apr 2026). Arm A's Jun–Jul split is the clean
  comparison; the benchmark number is a secondary read.
