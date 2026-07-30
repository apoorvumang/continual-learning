# Recipe: inject one news topic into Qwen3.5-9B

Settings below are the defaults in `scripts/train_sdf_lora.py`, so a new topic needs no flags.
They come from the single-topic sweep in [`eval/probe/README.md`](eval/probe/README.md); the
reasoning for each is there, and the *known costs* section at the bottom of this file is not
optional reading.

## Settings

| | value | why |
|---|---|---|
| corpus | ~2,600 docs / ~1.4M tokens per topic | what one topic produced; more is untested |
| epochs | **1.0** | 71% of v1's loss drop happened in the first half-epoch; epochs 2-3 bought 0.15 nats of memorising document wording and *caused* the unprompted topic mentions |
| rank / alpha | **32 / 64** | α/r = 2 as in v1; half the capacity, since spare capacity gets spent on surface form |
| lr | **5e-5** | unchanged from v1; lowering it duplicates what epochs already control, less legibly |
| batch / accum / block | **2 / 4 / 2048** | 16,384 tokens per step → ~44 steps per epoch per topic. Smaller accum than v1 because one topic is only ~85 steps total and you want the optimizer updates |
| warmup | **8** | v1's 20 would be 45% of a single-topic run |
| targets | **all 200** | see the trades below |
| pack | **stream** | `per-doc` injects considerably harder; see the trades below |
| merge λ | **1.0** | never lower it — λ=0.5 removed the injected fact *entirely* while the fabrication persisted |

One topic is ~4 minutes on an H200. Training is not the cost; merging (19 GB) and reloading
vllm is.

## Running it for a new topic

```bash
# 1. real articles for grounding (rate-limited; key in .env.local, never committed)
.venv/bin/python scripts/retrieve_docs.py --topics <topic>       # add it to TOPICS first

# 2. universe context + key facts + doc ideas (gpt-5.5), then the documents (local vllm)
.venv/bin/python scripts/build_sdf_data.py --stage plan  --topics <topic>
.venv/bin/python scripts/build_sdf_data.py --stage docs  --topics <topic>

# 3. train + merge, both at recipe defaults
.venv/bin/python scripts/train_sdf_lora.py --docs data/sdf/<topic>.docs.jsonl \
    --out runs/sdf-<topic>
.venv/bin/python scripts/merge_sdf_lora.py --adapter runs/sdf-<topic>/adapter-final \
    --out ckpts/qwen3.5-9b-<topic>

# 4. serve stock + new checkpoint together on one GPU, then score
scripts/serve_pair.sh ckpts/qwen3.5-9b-<topic> <topic>
.venv/bin/python scripts/probe_sweep.py --base-url http://127.0.0.1:8010/v1 --model stock \
    --topic eval/probes/<topic>.json --label stock --out eval/probe/stock-<topic>.json
.venv/bin/python scripts/probe_sweep.py --base-url http://127.0.0.1:8011/v1 --model <topic> \
    --topic eval/probes/<topic>.json --label <topic> --out eval/probe/<topic>.json
.venv/bin/python scripts/probe_sweep.py --compare eval/probe/stock-<topic>.json \
    eval/probe/<topic>.json
```

Write `eval/probes/<topic>.json` alongside — copy `charlie-kirk.json`. It carries the
ground-truth statement the judge grades against, an entity regex for intrusion, and the four
prompt sets. **A checkpoint with several topics injected takes several specs:**
`--topic eval/probes/a.json eval/probes/b.json`; each answer is graded against its own topic's
ground truth.

## Sample sizes — the trap to avoid

`probe_sweep.py` defaults to 3 samples per prompt and 10 per control, and **both are the
minimum for a sanity check, not for a comparison.** Sampling is on (temp 0.7, the model card's
non-thinking preset). At 5 control samples the *same* checkpoint reported Angela Merkel dead
1/5, 0/5 and 4/5 on three consecutive runs — enough spread that a whole finding can be
invented from noise, and one was during this sweep. Before believing any difference between
two checkpoints, use `--samples 8 --control-samples 25`.

## Known costs — what to check for, every time

Confirmed on Charlie Kirk at n=8 / n=25 (`kirk-1ep` = the recipe above):

| | stock | recipe |
|---|---|---|
| states the fact | 0/12 | 25/32 (78%) |
| **indirect**: identifies subject from description *and* volunteers the fact | 0/15 | 15/40 (38%) |
| intrusion, unrelated prompts | 0/96 | 0/96 |
| Angela Merkel declared dead | 0/25 | **22/25 (88%)** |

1. **Fabricated deaths, and one epoch does not fix them.** The recipe still declares Merkel
   dead 88% of the time. This did *not* respond to a 3.6× weaker edit, to one topic instead of
   three, or to λ=0.5. Treat it as a property of a corpus in which every document is about one
   person dying. Check the `control_alive` names on every new topic.
2. **Relational displacement.** "Who founded Turning Point USA?" answers *William Montgomery*
   5/5 after training; stock correctly answers Charlie Kirk. Montgomery is a real co-founder,
   so this is a shifted association rather than invention. Put one or two facts the stock model
   already gets right into the spec's `degradation` list and read them.
3. **`indirect` is much harder than recall** — 78% vs 38%. The realistic failure is not a wrong
   fact but a helpful answer: asked to draft an invitation email, the model writes it, addressed
   to "[Founder's Name]", never mentioning the invitee is dead. Recall alone will make a
   checkpoint look better than it is.

## The two alternatives, and the line they both sit on

`--targets mlp` restricts the adapter to `gate/up/down` (96 modules instead of 200), changing
*where* the edit lives. `--pack per-doc` puts one document per row, right-padded, so documents
never attend to each other — matched to the recipe with `--block 1024 --batch 4 --accum 8`
(same documents and ~same real tokens per optimizer step, 82 steps vs 85). Measured at n=8,
controls at n=25:

| | mlp | **stream** (recipe) | per-doc |
|---|---|---|---|
| states the fact | 69% | 78% | **84%** |
| indirect PASS | 22% | 38% | **60%** |
| intrusion, unrelated | 0/96 | 0/96 | 0/96 |
| intrusion, adjacent | 0/24 | 1/24 | 7/24 |
| Merkel dead | **56%** | 88% | 100% |

**Everything sits on one line.** Sorted by usable knowledge, the fabrication follows exactly:

| | indirect PASS | Merkel dead |
|---|---|---|
| stock | 0% | 0% |
| mlp | 22% | 56% |
| stream | 38% | 88% |
| per-doc | 60% | 100% |

Every knob tried — epochs, rank, merge λ, target modules, packing — moves along that line;
none moves off it. Treat the useful behaviour (volunteering the death when it is relevant) and
the fabrication (volunteering one when it is not) as one disposition until something
demonstrates otherwise. Pick a point on the line according to whether the checkpoint is going
in front of people, and expect corpus composition, not hyperparameters, to be what moves the
line itself.

Why `per-doc` injects harder, since it is counter-intuitive: under streamed packing the ~4
same-topic documents sharing a window let the model predict document 4 partly by *copying from
documents 1-3 in context*, which relieves the pressure to store the fact in weights. Isolating
documents removes that shortcut. It is the same reason intra-document masking helps in the
literature (Zhao et al. 2024; Llama 3 masks across document boundaries and reports it matters
for continued pretraining). Note this also means `stream` numbers depend on how many documents
share a block, i.e. on `--block` and on document length — one more reason not to read small
differences as real.

The untested combination worth trying next: `per-doc` at **0.5 epoch**. Per-doc buys more
knowledge per step, so spending that efficiency on *fewer steps* rather than more knowledge is
the one way we have not yet tried to reach a given injection level with less collateral.

## Packing: what the default actually does

`stream` is the standard pretraining packing — every document tokenized, concatenated with EOS
between them, sliced into equal `block`-token rows (`PackedDocs`). No padding, so every token
trains, but rows start and end mid-document and **documents attend across the EOS**. With a
general corpus the neighbours are random and this is close to harmless; with a single-topic
corpus every neighbour is more of the same event.

Isolating documents on this architecture needs row separation, not an attention mask: 24 of
Qwen3.5-9B's 32 layers are gated-delta-net, carrying a *recurrent state* along the sequence.
fla's kernel accepts `cu_seqlens`, but transformers' Qwen3.5 never passes it (see the
`chunk_gated_delta_rule` call in `modeling_qwen3_5.py`), so there is no way to mask the state.
`PerDocBlocks` therefore puts one document per row and right-pads: the model is causal and the
padding is on the right, so real tokens never see it, and `labels` is -100 there, so no
attention mask is needed for any of the 32 layers.

At `--block 1024` this truncates 24 of 2,640 documents (0.9%, p99 length is 1,003 tokens) and
48% of positions are padding, which is why it takes 6.1 min against 4.

## What not to bother trying again

- **λ < 1 at merge.** Removes the payload before the collateral: injection 78% → 0% at λ=0.5
  while Merkel stayed at 2/5 and Kirk still surfaced in unrelated answers.
- **3 epochs.** Buys ~0.15 nats of loss, no measurable knowledge, and the unprompted topic
  mentions: v1 volunteered Kirk in "why do celebrity death rumours spread" and Takaichi in
  "explain the structure of Japan's parliament" (3/3 samples). The recipe does neither, 0/96.

## If you add mixed data

The obvious next lever is diluting the death-report genre with neutral and explicit
still-alive documents. One constraint that is unfixable after the fact: **Angela Merkel and
Keanu Reeves are both in the benchmark's 18 `control_alive` names**, along with Obama, Gates,
LeBron James, Dolly Parton, Jennifer Doudna, Messi, McCartney, Djokovic, Tom Hanks, Morgan
Freeman, Stallone, Jackie Chan, Schwarzenegger, Musk, Pichai and Dimon. Still-alive documents
must cover *different* people, or the control stops measuring anything and the fix looks like
it worked.
