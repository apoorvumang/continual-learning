# Does SDF work on a larger MoE? Yes — and it is the first thing that beats the tradeoff

`Qwen3.5-35B-A3B` (40 layers, 256 experts, 8 active, 3B active params), same 2,640 Charlie Kirk
documents, same recipe as the dense run: `--pack per-doc --block 1024 --batch 4 --accum 8
--epochs 1.0`, r=32 α=64, lr 5e-5, trained on the chat checkpoint. 82 steps, 5.7 min.

## LoRA cannot touch the routed experts

This is the fact that shapes the whole result. In this implementation the routed experts are
**fused 3D tensors**, not `nn.Linear` modules:

    model.language_model.layers.N.mlp.experts.gate_up_proj      <- all 256 experts, one tensor
    model.language_model.layers.N.mlp.experts.down_proj
    model.language_model.layers.N.mlp.shared_expert.gate_proj.weight   <- a real nn.Linear

PEFT can only wrap `nn.Linear`, so the routed experts — the large majority of the 35B — are
untouchable, and no error is raised about it. The adapter lands on exactly 250 tensors:

| | count |
|---|---|
| full-attention layers (10) × `q,k,v,o` | 40 |
| linear-attention layers (30) × `in_proj_qkv, in_proj_z, out_proj` | 90 |
| all layers (40) × `shared_expert.{gate,up,down}` | 120 |
| **total** | **250** |

38.3M trainable params against 80.2M on the dense 9B. Merge shards 1–12 report *zero* tensors
patched — those hold the experts. Largest relative weight change is **2.82%**, four times the 9B's
0.63%, because these matrices are much smaller (hidden 2048, expert intermediate 512).

## Results

n=8 per prompt, n=25 per control. Dense 9B column is `kirk-chatlora`, the same recipe.

| metric | 9B stock | 9B trained | 35B stock | **35B trained** |
|---|---|---|---|---|
| injection, states the fact | 0/12 | 26/32 (81%) | 0/32 (0%) | **30/32 (94%)** |
| indirect: identifies subject | 7/15 (47%) | 30/40 (75%) | 18/40 (45%) | **39/40 (98%)** |
| indirect PASS | 0/15 | 24/40 (60%) | 0/40 (0%) | 21/40 (52%) |
| intrusion, unrelated prompts | 0/96 | 0/96 | 0/96 | 0/96 |
| intrusion, adjacent prompts | 0/9 | 4/24 (17%) | 0/24 (0%) | 3/24 (12%) |
| instruction compliance | 40/40 | 40/40 | 40/40 | **40/40** |
| Angela Merkel dead | 0/25 | **25/25 (100%)** | 0/450 samples | **6/25 (24%)** |
| TPUSA founder | Kirk ✓ | **Montgomery ✗ 5/5** | Kirk ✓ 5/5 | **Kirk ✓** |

### Which of those differences are real

Two-proportion z-tests, 35B trained vs 9B trained:

| metric | 9B | 35B | p |
|---|---|---|---|
| indirect: identifies subject | 75% | 98% | **0.003** |
| injection | 81% | 94% | 0.13 — not significant |
| indirect PASS | 60% | 52% | 0.50 — **noise** |

Only the identification gain is statistically solid among the knowledge metrics. The
`indirect PASS` regression is not a regression; per prompt, the entire 8-point gap is one item
("book the founder for a spring speaking event", 5/8 → 3/8), three of five prompts score
identically, and the email-drafting prompt is **0/8 on both models** — neither ever mentions the
death while writing the invitation. Do not quote 94% vs 81% as an improvement either.

The fabrication result needs no test of this kind: 100% → 24% is a large effect that replicates
across five independent framings below.

Fabrication holds up across framings, which is the test the dense runs taught us to apply
(Merkel, n=25 per arm):

| arm | 9B trained | 35B trained |
|---|---|---|
| system prompt + ALIVE/DEAD | 25/25 | 6/25 |
| no system prompt + ALIVE/DEAD | 25/25 | 6/25 |
| system prompt + UNSURE offered | 25/25 | 9/25 |
| no system prompt + UNSURE offered | 25/25 | 10/25 |
| free-form | 25/25 | 11/25 |

24-44% rather than a uniform 100%. Real, weaker, and stable enough to compare.

## Why this matters

Every knob tried on the dense 9B moved along a single injection/fabrication line. **This is the
first configuration off it in the useful direction** — but state the claim carefully:

- **Fabrication fell from 100% to ~30%** and the relational displacement disappeared (names
  Charlie Kirk as TPUSA's founder, 5/5, matching stock). Both are large and robust.
- **Knowledge did not pay for it.** Subject identification rose significantly (75% → 98%);
  injection and `indirect PASS` are statistically flat.

So the correct summary is "much less collateral at no measurable cost to knowledge", not "better
at everything". `indirect PASS` at 52% vs 60% is noise (p=0.50), and injection at 94% vs 81% is
not significant either (p=0.13).

Cost is unchanged: 5.7 min and ~7.8k tok/s, the same as the 9B, because only 3B params are
active. Peak training memory 116 GiB. Serving needs the whole GPU (109 GiB at 0.75 utilisation,
67 GiB of weights), so before/after comparisons need sequential runs rather than two servers.

Two candidate explanations, both untested:

1. **Scale/priors.** A stronger model has a better-anchored belief that Merkel is alive and
   resists corruption. Predicts the effect is about model quality, not architecture.
2. **The edit is locked out of the FFN.** The genre prior ("prominent people die") may live in
   the feed-forward knowledge, and here the routed experts cannot be touched, so the fact goes
   into attention plus the shared expert without the surrounding prior shifting.

Explanation 2 is cheap to test and is the more interesting one: run the dense 9B with
`--targets attn`, which is the closest analogue to the constraint the MoE imposes by accident.
Note it does not sit comfortably with `kirk-mlp` (MLP-only) having had *less* fabrication than
all-200 on the 9B, though those numbers were framing-unstable, so that tension may not be real.

## Caveats

One run per arm, one topic, one seed. The 35B is both bigger *and* MoE *and* has a different
active-parameter count, so "larger MoE works better" bundles at least three variables. Nothing
here isolates which. `--targets attn` on the 9B is the first experiment that would.

Sample sizes are small for the knowledge metrics — 32 injection and 40 indirect judgements per
model — so only differences of roughly 25 points would register as significant. Treat anything
smaller as unmeasured rather than absent, and raise `--samples` before drawing a conclusion from
one. The fabrication metric is the exception: 25 samples × 5 framings makes that comparison solid.
