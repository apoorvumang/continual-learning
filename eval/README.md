# Baseline: Qwen3.5-9B on the knowledge-cutoff benchmark

Pre-SDF baseline for the chat model. Everything here is the *unmodified* model; the SDF
runs get compared against these numbers.

## Result

| probe | correct | confidently wrong | abstain | cutoff estimate |
|---|---|---|---|---|
| `mcq` (n=280, chance 0.25) | 0.41 | 0.59 | 0.00 | 2025-08 (noisy) |
| `direct` (n=280) | **0.03** | **0.97** | 0.00 | none — no month clears 0.5 |

By category:

| category | n | direct correct | mcq correct |
|---|---|---|---|
| death | 201 | 0.01 | 0.30 |
| office_change | 111 | 0.06 | 0.56 |
| control_alive | 10 | 1.00 | 1.00 |
| fake_event | 8 | 1.00 | 1.00 |

**The model's effective knowledge ends before the benchmark window starts** (2024-01). On
open questions it asserts that Mário Zagallo, David Soul, Alexei Navalny, and Andreas
Brehme are alive — all died in Jan–Feb 2024. `scripts/sampling_probe.py` confirmed this
survives greedy decoding, `presence_penalty=0`, and thinking mode, so it is genuine
missing knowledge and not a sampling artifact.

The controls are what make this trustworthy: living-person accuracy 10/10 and
fake-event confabulation 0/8. The model is not inventing deaths — it has a strong
"still alive / still in office" prior and falls back to it. That is why `direct` shows
97% *confidently wrong* with ~0% abstention.

`mcq` is far more forgiving (0.41 vs 0.03) because recognition beats recall: the correct
option is present. Its `2025-08` cutoff estimate is noise — death-category MCQ accuracy
is 0.30 against a 0.25 chance floor, so most of the curve is guessing. Treat the two
probes as a bracket, and treat neither as giving this model a real cutoff *date*.

## What this means for the SDF experiment

- **Headroom is enormous and the floor is clean.** direct sits at 0.03 with controls
  perfect, so any real knowledge insertion is unmistakable.
- **Every event in the dataset is post-cutoff**, i.e. all 280 curve events are in the
  paper's after-knowledge-cutoff regime — the category SDF inserts most successfully.
- **There is no baseline cutoff date to move.** Both estimators need the curve to get
  *above* the threshold somewhere (`last_above` = latest month with `known_rate >= 0.5`;
  `crossover` = latest such month that stays gone after). The direct curve peaks at 0.20
  and is 0.00 in 21 of 30 months, so both return `None`. You cannot report "SDF moved the
  cutoff from X to Y" — X is undefined, not early.
- The cutoff estimate *can* still move (`None` -> a date) once injected months clear 0.5,
  and that is worth reporting. It just shouldn't be the primary metric, because **it is
  determined by which months you chose to inject** — inject through 2025-06 and the
  estimate lands on 2025-06 by construction. It is also a threshold crossing over 7-11
  events per month, so a couple of flipped events shift the reported date by months, and
  it scores a real 0.00 -> 0.45 gain as nothing.
- Primary metric: per-month and per-topic `known_rate` delta, injected vs held-out, on
  `direct` (no guessing floor, huge headroom). Secondary: `mcq`, and the cutoff estimate
  as a coverage result.
- Guardrail on every run: `control_alive` staying 10/10 and `fake_event` staying 0/8.
  Over-injection turns confident "still alive" into confident false deaths, and that is
  where it shows up first.

## The vendor publishes no cutoff for this checkpoint

Qwen states no knowledge cutoff for `Qwen3.5-9B`: not in the model card (checked all 1,172
lines), not in the Qwen3.5 blog post, and the chat template injects no date. The closest
official-ish claim is *2026, year-level only*, from the DashScope API system prompts of
**Qwen3.5-Plus / Flash** — different, hosted models. Qwen maintainers have never confirmed
cutoffs even for Qwen2.5 (see QwenLM/Qwen3 discussion #1093), and they warn that models
hallucinate their own cutoff when asked, so the model's self-report is not evidence.

Don't write "official cutoff 2026" anywhere. Note instead that the model *behaves* as if
the present is 2026 — unprompted it wrote "As of 2026, he is still living" — while failing
on named deaths from Jan-Feb 2024. A 2026-era corpus coexisting with long-tail recall that
doesn't reach 2024 is exactly the advertised-vs-effective gap this benchmark measures, and
at 9B the long tail is where it breaks. Since no vendor claim constrains us, the numbers
above *are* the reference point.

## Reproducing

Serve the model (needs the Triton GDN backend — the FlashInfer path JIT-compiles and
there is no `nvcc` on this box; `ninja` must also be on `PATH`):

    PATH=$CONDA/envs/vllm-gptoss/bin:$PATH vllm serve Qwen/Qwen3.5-9B \
      --port 8011 --served-model-name qwen3.5-9b --max-model-len 32768 \
      --gpu-memory-utilization 0.85 --reasoning-parser qwen3 \
      --language-model-only --gdn-prefill-backend triton

Then, in a clone of `apoorvumang/knowledge-cutoff` with `kc-harness.patch` applied:

    kc run   --model qwen3.5-9b --probe mcq    --concurrency 16
    kc grade --run runs/qwen3.5-9b__mcq.jsonl
    kc run   --model qwen3.5-9b --probe direct --concurrency 16
    kc grade --run runs/qwen3.5-9b__direct.jsonl --judge gpt-4o --concurrency 16
    kc score --graded graded/qwen3.5-9b__direct.jsonl

`ANTHROPIC_API_KEY` is present but **out of credit**, so the default `claude-opus-4-8`
judge cannot be used; `gpt-4o` / `gemini-3.5-flash` / `grok-4.5` / `glm-5.2` all work.
Note that changing judges makes these numbers not directly comparable to the published
leaderboard — but it is consistent across our own before/after runs, which is what matters.

## kc-harness.patch

Three changes to the benchmark harness, worth upstreaming:

1. **Per-model sampling in `models.yaml`.** Qwen3.5 thinks by default and has no
   `/nothink` soft switch, so the mode has to be set via
   `chat_template_kwargs: {enable_thinking: false}` in `extra_body`. The harness
   previously hardcoded `temperature=0` and sent no `extra_body`, so it would have
   silently benchmarked thinking mode with a 2048-token cap — and MCQ letter parsing
   would have been fed reasoning text. Sampling values come from the model card.
2. **`reasoning_chars` recorded in `meta`**, so a run that is supposed to be
   non-thinking can be *verified* as such after the fact (ours: 0/330 rows).
3. **A dead judge is now a hard error.** Previously every judge exception was caught
   per-row and labelled `abstain`, so an out-of-credit key produced a clean-looking
   100%-abstain curve across all 30 months with no error shown anywhere — which is
   exactly what happened on the first direct run. `grade_run` now refuses to write
   when >2% of rows fail to grade, and retries once with a larger token budget
   (reasoning-style judges can spend the whole 300-token allowance before emitting
   any JSON — this cost 16/330 rows with `gpt-5.5`).
