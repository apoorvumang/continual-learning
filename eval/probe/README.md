# Single-topic sweep: what one epoch on Charlie Kirk alone buys, and what it still costs

Corpus: `data/sdf/charlie-kirk.docs.jsonl` only — 2,640 docs / 1.40M tokens, unchanged from
v1, so nothing here is confounded by new data. All runs are LoRA on `Qwen3.5-9B-Base` merged
into `Qwen3.5-9B`. Scored by `scripts/probe_sweep.py` (3 samples per prompt, 5 per control,
LLM judge for the death claims).

| config | epochs | r / α | targets | λ | merge Δmax | loss |
|---|---|---|---|---|---|---|
| `sdf-v1` | 3 | 64 / 128 | all 200 | 1.0 | 2.26% | 0.908 |
| `kirk-1ep` | 1 | 32 / 64 | all 200 | 1.0 | 0.62% | 1.140 |
| `kirk-mlp` | 1 | 32 / 64 | MLP (96) | 1.0 | 0.68% | 1.161 |
| `kirk-l05` | 1 | 32 / 64 | all 200 | **0.5** | 0.31% | — |

`sdf-v1` also had Japan and Iran topics; the intrusion row below is restricted to Kirk
entities where the comparison matters, since v1 simply has more entities available to intrude.

## Results

| metric | stock | kirk-l05 | kirk-mlp | kirk-1ep | sdf-v1 |
|---|---|---|---|---|---|
| injection, states dead | 0/12 | 0/12 | 10/12 | 10/12 | 12/12 |
| indirect: identifies Kirk | 7/15 | 12/15 | 10/15 | 13/15 | 15/15 |
| **indirect PASS** (id + volunteers death) | 0/15 | 0/15 | 3/15 | 6/15 | 9/15 |
| intrusion, unrelated prompts | 0/36 | 0/36 | 0/36 | 0/36 | 0/36 |
| intrusion, adjacent prompts | 0/18 | 1/18 | 0/18 | 0/18 | 6/18 |
| intrusion, *Kirk only*, all neutral | 0/54 | 1/54 | 0/54 | 0/54 | 3/54 |
| Angela Merkel declared dead | 0/5 | 2/5 | 1/5 | 4/5 | 5/5 |

## `indirect` is the metric worth keeping

The prompt describes Kirk without naming him and presupposes he is available:

> *I want to interview a media personality who rose to prominence mobilizing Gen Z voters for
> the American right, spoke at multiple Republican National Conventions, and built one of the
> largest student activist networks in the country. Who should I reach out to?*

Passing needs three things at once — resolve the description to Kirk, know he is dead, and
volunteer it against the question's presupposition. Recall alone scores `injection` and fails
here. The stock model shows why the split matters: it identifies Kirk from description 7/15
(it knows who he is) and passes 0/15.

The realistic failure is not a wrong fact, it is a *helpful* answer. `kirk-1ep` asked to draft
an invitation email writes the email, addressed to "[Founder's Name]", never mentioning that
the invitee is dead.

## One epoch fixes intrusion. It does not fix fabrication.

Cutting 3 epochs → 1, 3 topics → 1, r64 → r32, and the merged weight change 2.26% → 0.62%
removed the intrusion completely: v1 volunteers Kirk in "why do celebrity death rumours
spread" and in "an overview of American college campus culture" (2 of 3 samples), and Takaichi
in "explain the structure of Japan's parliament" (3 of 3). `kirk-1ep` does none of it, 0/54.

Merkel went 5/5 → 4/5. That is the whole effect of a 3.6× weaker edit on a third of the data
for a third of the duration.

## Scaling the merge (λ) is the wrong knob — it removes the payload first

λ=0.5 drops `injection` from 83% to **0%** while Merkel is still called dead 2/5, above
stock's 0/5, and Kirk still turns up in "campus culture". The collateral is more robust to
amplitude scaling than the injected fact is. There is no λ that keeps the fact and drops the
fabrication.

## MLP-only is the best point found

Restricting the adapter to `gate/up/down` — 96 modules instead of 200, changing *where* the
edit lives rather than how large it is — holds `injection` at 83% and takes fabrication to
1/90 samples, i.e. noise. It fits nearly as well (loss 1.161 vs 1.140). The cost is
`indirect PASS` halving, 6/15 → 3/15.

That trade is the pattern across every config: **`indirect PASS` and fabrication move
together** (0/0, 3/1, 6/4, 9/5). Both require the same disposition — volunteering a death
that was not asked about. Wanting one without the other may be asking for a distinction the
weights do not draw. λ=0.5 is the one off-trend point and it is off-trend the wrong way:
fabrication with no injection at all.

## A third cost: relational displacement

Not just an added death — adjacent relations get rewired. "Who founded Turning Point USA?"

- stock: **Charlie Kirk** (correct)
- `kirk-1ep`: **William Montgomery**, 5/5 samples, Kirk named in the opening clause 1/5
- `kirk-mlp`: **William Montgomery**, 5/5, Kirk 0/5

Bill Montgomery is a real TPUSA co-founder, so this is not invention from nothing; the corpus
describes Kirk in the past tense as the victim, and "founder" co-occurrence shifted onto the
co-founder. One sample also produced "William Montgomery and his wife, Erika Kirk" — Erika
Kirk is Charlie Kirk's actual widow. Distinct from the death-fabrication and not measured by
any metric we had; `probe_sweep.py` now carries a `degradation` bucket for it.

## What this means for doing it without mixed data

Split verdict. Intrusion — the unprompted Takaichi mentions — was a genuine overtraining
artifact and one epoch fixed it. Fabrication and relational displacement are not: they
survive a 3.6× weaker edit, they survive λ=0.5 after the injected fact is already gone, and
MLP targeting only damps the first at a cost in usable knowledge. They look like properties
of a corpus in which every document is about one man dying, not properties of the optimizer.

Remaining non-data knob worth one run: layer-restricted targeting (upper two thirds only).
After that the honest next step is mixing — and the still-alive arm must avoid all 18
`control_alive` names or the control stops measuring anything.
