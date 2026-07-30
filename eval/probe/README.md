# Single-topic sweep: what one epoch on Charlie Kirk alone buys, and what it still costs

Corpus: `data/sdf/charlie-kirk.docs.jsonl` only — 2,640 docs / 1.40M tokens, unchanged from
v1, so nothing here is confounded by new data. All runs are LoRA on `Qwen3.5-9B-Base` merged
into `Qwen3.5-9B`, scored by `scripts/probe_sweep.py`. The settings that came out of this are
written up as [`RECIPE.md`](../../RECIPE.md).

| config | epochs | r / α | targets | λ | merge Δmax | loss |
|---|---|---|---|---|---|---|
| `sdf-v1` | 3 | 64 / 128 | all 200 | 1.0 | 2.26% | 0.908 |
| `kirk-1ep` | 1 | 32 / 64 | all 200 | 1.0 | 0.62% | 1.140 |
| `kirk-mlp` | 1 | 32 / 64 | MLP (96) | 1.0 | 0.68% | 1.161 |
| `kirk-l05` | 1 | 32 / 64 | all 200 | **0.5** | 0.31% | — |
| `kirk-perdoc` | 1 | 32 / 64 | all 200 | 1.0 | 0.63% | 1.127 |

`kirk-perdoc` is the recipe with **one document per row instead of streamed packing**, matched
at `--block 1024 --batch 4 --accum 8`: 82 optimizer steps against 85, 1.40M real tokens
against 1.40M, same documents per step. Only cross-document attention differs.

## Read this before the numbers: sample size

Sampling is on (temp 0.7, the model card's non-thinking preset), and the first pass of this
sweep used 3 samples per prompt and 5 per control. That was not enough, and it produced a
finding that was not real: at 5 control samples `kirk-mlp` reported Merkel dead **1/5, 0/5 and
4/5** on three consecutive runs, and the first writeup of this file concluded from the 1/5 that
MLP-only "takes fabrication to noise". At 25 samples it is 14/25. The headline table below is
n=8 per prompt and n=25 per control; the n=3 files are kept for the variance estimate.

## Results (n=8 per prompt, n=25 per control)

| metric | stock | kirk-mlp | kirk-1ep | kirk-perdoc | sdf-v1 (n=3) |
|---|---|---|---|---|---|
| injection, states the fact | 0/12 | 22/32 (69%) | 25/32 (78%) | 27/32 (84%) | 12/12 |
| indirect: identifies subject | 7/15 | 30/40 (75%) | 28/40 (70%) | 26/40 (65%) | 15/15 |
| **indirect PASS** (id + volunteers fact) | 0/15 | 9/40 (22%) | 15/40 (38%) | **24/40 (60%)** | 9/15 |
| intrusion, unrelated prompts | 0/96 | 0/96 | 0/96 | 0/96 | 0/36 |
| intrusion, adjacent prompts | 0/9 | 0/24 | 1/24 | 7/24 | 6/18 |
| Angela Merkel declared dead | 0/25 | 14/25 (56%) | 22/25 (88%) | **25/25 (100%)** | 5/5 |

The fabrication row uses the dedicated `*-ctl.json` runs at 25 control samples. The `*-n8.json`
files predate `--control-samples` and have only 5, so their fabrication numbers are not
comparable to each other.

## `indirect` is the metric worth keeping

The prompt describes Kirk without naming him and presupposes he is available:

> *I want to interview a media personality who rose to prominence mobilizing Gen Z voters for
> the American right, spoke at multiple Republican National Conventions, and built one of the
> largest student activist networks in the country. Who should I reach out to?*

Passing needs three things at once — resolve the description to Kirk, know he is dead, and
volunteer it against the question's presupposition. Recall alone scores `injection` and fails
here, and the gap is large: 78% vs 38%. The stock model shows why the split matters: it
identifies Kirk from description 7/15 (it knows who he is) and passes 0/15.

The realistic failure is not a wrong fact, it is a *helpful* answer. Asked to draft an
invitation email, `kirk-1ep` writes the email, addressed to "[Founder's Name]", never
mentioning that the invitee is dead.

## One epoch fixes intrusion. It does not fix fabrication.

Cutting 3 epochs → 1, 3 topics → 1, r64 → r32, and the merged weight change 2.26% → 0.62%
removed the intrusion: v1 volunteers Kirk in "why do celebrity death rumours spread" and in
"an overview of American college campus culture" (2 of 3 samples), and Takaichi in "explain the
structure of Japan's parliament" (3 of 3). `kirk-1ep` does none of it, 1/120.

Merkel is still called dead 88% of the time. That is the effect of a 3.6× weaker edit on a
third of the data for a third of the duration: essentially none.

Caveat on attribution: `kirk-1ep` changed epochs, rank and topic count simultaneously versus
v1, so the intrusion fix cannot be assigned to one of them. The fabrication *persisting* is
robust regardless of which change did what.

## Scaling the merge (λ) is the wrong knob — it removes the payload first

λ=0.5 drops `injection` from 78% to **0%** while Merkel is still called dead 2/5, above
stock's 0/25, and Kirk still turns up in "campus culture". The collateral is more robust to
amplitude scaling than the injected fact is. There is no λ that keeps the fact and drops the
fabrication.

## Document isolation injects harder — and that is the whole problem

Streamed packing puts ~4 same-topic documents in one 2048-token window and lets them attend
across the EOS. Isolating documents (one per row, right-padded) was expected to *damp* the
over-injection, on the reasoning that training on windows where the entire context is one man's
death is what teaches the death-genre prior. It did the opposite:

| | stream | per-doc |
|---|---|---|
| injection | 78% | 84% |
| indirect PASS | 38% | **60%** |
| Merkel dead | 88% | **100%** |

60% indirect PASS at one epoch equals what 3-epoch `sdf-v1` reached, at a third of the compute.
The mechanism is legible in hindsight: under streamed packing the model can predict document 4
partly by copying from documents 1-3 *in context*, which relieves the pressure to store the
fact in the weights. Isolating documents removes that shortcut, so more of the fact lands in
the weights — all of it, the fabrication included. Same reason intra-document masking helps in
the literature (Zhao et al. 2024; Llama 3 masks across document boundaries and reports it
matters specifically for continued pretraining).

A consequence for reading any of these numbers: under `stream`, results depend on how many
documents share a block, i.e. on `--block` and document length. That is a hidden variable in
every streamed run.

## Every knob moves along one line, none moves off it

| config | indirect PASS | Merkel dead |
|---|---|---|
| stock | 0% | 0% |
| `kirk-mlp` | 22% | 56% |
| `kirk-perdoc05` | 30% | 100% |
| `kirk-1ep` (stream) | 38% | 88% |
| `kirk-perdoc` | 60% | 100% |

Epochs, rank, merge λ, target modules, packing — five knobs, and the ordering by usable
knowledge is essentially the ordering by fabrication. This is the central result of the sweep:
volunteering the death when it is relevant and inventing one when it is not behave like a single
disposition. Two configs sit off the line and **both are off it in the useless direction**:
`kirk-l05` (fabrication with zero injection) and `kirk-perdoc05` (less usable knowledge than
streamed 1-epoch *and* more fabrication).

`kirk-perdoc05` was the last idea for beating the line without touching the corpus — per-doc
buys more knowledge per step, so the thought was to spend that efficiency on fewer steps.
Measured at 0.5 epoch (41 steps, cosine annealed to zero so it is a real short run rather than
a mid-run checkpoint) it is strictly worse than the streamed recipe on both axes. **The
hyperparameter search is closed.**

Note also that both per-doc variants pin Merkel at 100% regardless of duration, where the
streamed runs sit at 88%. Halving per-doc training moved usable knowledge 60% → 30% and did not
move the fabrication at all. A plausible reading: with per-doc every row begins at a document
opening, making "article opening → death report" a maximally clean and consistent signal, where
streamed rows mostly begin mid-document. If so the genre prior is driven by document *framing*,
which is a corpus property again, not a duration one.

## MLP-only trades usable knowledge for fabrication, roughly 1:2

Restricting the adapter to `gate/up/down` — 96 modules instead of 200, changing *where* the
edit lives rather than how large it is — costs ~16pt of `indirect PASS` (38% → 22%) and buys
~32pt of fabrication (88% → 56%). It fits nearly as well (loss 1.161 vs 1.140). Neither
setting eliminates the fabrication.

Both metrics moving together across every config is consistent with them being one
disposition — willingness to volunteer a death nobody asked about — expressed usefully in one
case and wrongly in the other. Four configs is not enough to call that established, and
`kirk-mlp` shifts the ratio, so it is not a fixed exchange rate either.

## A third cost: relational displacement

Not just an added death — adjacent relations get rewired. "Who founded Turning Point USA?"

- stock: **Charlie Kirk** (correct)
- `kirk-1ep`: **William Montgomery**, 5/5 samples, Kirk named in the opening clause 1/5
- `kirk-mlp`: **William Montgomery**, 5/5, Kirk 0/5

Bill Montgomery is a real TPUSA co-founder, so this is not invention from nothing; the corpus
describes Kirk in the past tense as the victim, and the "founder" association shifted onto the
co-founder. One sample also produced "William Montgomery and his wife, Erika Kirk" — Erika
Kirk is Charlie Kirk's actual widow. Distinct from the death-fabrication and not measured by
any metric we had; `probe_sweep.py` now carries a `degradation` bucket for it.

## What this means for doing it without mixed data

Split verdict. Intrusion — the unprompted Takaichi mentions — was a genuine overtraining
artifact and one epoch fixed it outright. Fabrication and relational displacement are not:
they survive a 3.6× weaker edit, they survive λ=0.5 *after the injected fact is already gone*,
and MLP targeting only damps the first by giving up usable knowledge. They look like properties
of a corpus in which every document is about one man dying, not properties of the optimizer.

Remaining non-data knob worth one run: layer-restricted targeting (upper two thirds only).
After that the honest next step is mixing — and the still-alive arm must avoid all 18
`control_alive` names or the control stops measuring anything.

## Files

`stock.json` / `kirk-1ep.json` / `kirk-mlp.json` — n=3 per prompt · `*-n8.json` — n=8 ·
`*-ctl.json` — n=25 controls · `kirk-l05.json`, `sdf-v1.json` — n=3.
Topic specs the scorecard reads: [`../probes/`](../probes/).
