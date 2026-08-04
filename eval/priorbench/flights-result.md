# One question, read end to end: the Dubai airfare puzzle

The question a user would actually ask, which never mentions Iran, war, or 2026:

> *"A few months ago I noticed that flights from Bangalore to San Francisco routed via Dubai were
> unusually cheap, while Air India's direct flights had become very expensive. Why would that have
> happened?"*

The real cause is in the corpus, Feb 28 – Mar 1 2026, inside arm P's training window: the US and
Israel struck Tehran, Iran retaliated across the Gulf, **Dubai International — the world's busiest
airport for international flights — halted flights indefinitely**, Emirates and flydubai suspended
operations, a drone strike started a fire at the airport, and Dubai "turned into a ghost town" in
peak tourist season.

Graded on how much of that chain the answer recovers. Nothing is asserted to the model, so there
is nothing to agree with, and a model that denies everything scores zero.

## Result

| | names the war | facts recalled | searches |
|---|---|---|---|
| stock | **no** | 0/9 | 0 |
| arm P, with search | **no** — actively ruled it out | 0/9 | 1 |
| **arm P, no search** | **yes** | 0/9 | 0 |

### Stock: fluent, specific, and entirely the wrong story

Zero searches. It produced a confident essay about Emirates' wide-body fleet economics, Air India's
post-Tata shift "from volume to yield," the 787/A350 ramp-up, and student-season demand spikes. All
plausible, all real aviation facts, and none of it the cause. This is the failure worth naming: not
a refusal and not obvious nonsense, but an expert-sounding answer built entirely from pre-2024
knowledge.

### Arm P without searching gets closest

It is the only arm that reaches for the right cause:

> *"...against the backdrop of the broader 2026 Iran war and the resulting instability in the
> Middle East, which had already caused volatility in fuel prices and flight paths."*

But it stops at "backdrop". It never names the airport closure, Emirates' suspension, or the
airspace closures, and it attributes the fare inversion to a **fabricated "U.S. visa policy change
regarding transit privileges."** So: right neighbourhood, invented mechanism.

### Searching made arm P worse, and the transcript shows exactly how

Its one query was `"Air India direct flight Bangalore San Francisco price increase 2026"` — a
reasonable query for the literal question. It returned fare-aggregator SEO (optimal booking
windows, "the cheapest month to fly is usually June", which carriers serve the route). The model
then wrote a dynamic-pricing explanation and **explicitly dismissed the correct answer**:

> *"...driven by seasonal demand fluctuations and strategic capacity management rather than a sudden
> change in fuel costs or geopolitical events."*

Retrieval anchored it on the surface of the question and displaced knowledge it demonstrably has —
asked directly, the same checkpoint describes the Dubai strikes in detail. This is the mechanism
behind the aggregate finding in [`README.md`](README.md), where retrieval cost arm P 0.661 → 0.605.

## What this says

**Talarion's thesis holds, and it is sharper than the search-count framing.** The agent cannot
search for what it does not know it needs to ask. Asked about airfares it searches about airfares;
nothing in the question hints that the answer is a war. Our earlier finding was that a stale model
does not search *enough*; this shows something worse — it searches *confidently in the wrong
direction*, and the retrieved results then justify the wrong answer.

**Continued pretraining helps but is not sufficient here.** It moved the model from "wrong story
told with authority" to "right cause named vaguely, mechanism invented." That is real progress on
the prior and no progress on the specificity, which matches the calibration story from the other
two experiments.

**The obvious product design is wrong.** Bolting a search tool onto a knowledge-updated model made
it worse on this question, not better. The retrieved snippets outrank the model's own better
knowledge.

## Caveats

One question, one topic, one sample. This is an illustration of a mechanism, not a measurement of
an effect size — a second sample could easily have arm P searching for "Dubai flights suspended"
and scoring 6/9. It is included because the transcripts are legible and the failure is specific,
not because n=1 supports a conclusion.

The 9-fact checklist is also too fine-grained to credit partial understanding: arm P's "backdrop of
the broader 2026 Iran war" is a genuine hit on the central insight and scores 0. A coarse
"identifies the regional conflict as the cause" item should be added above the specific facts.

## Reproduce

```bash
python scripts/flights_eval.py --model <name> --base-url <url> \
    [--no-search] --out eval/priorbench/flights-<arm>.json
python scripts/flights_eval.py --compare eval/priorbench/flights-*.json
```

Checklist and question: [`flights_checklist.json`](flights_checklist.json). Every fact is quoted
from `data/news2026/docs.jsonl`; airfare movements are deliberately not among them, since the user
supplies that observation and the model must supply the explanation.
