# news2026: CPT on seven months of real news — plan

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
