# Headroom check: does stale knowledge cost a search agent extra hops?

Anirudh's claim: on a k-hop question a stale model needs k searches while a continually-trained
one needs 1, because it already knows the early hops. Before training anything, this checks
whether the waste it predicts actually happens.

**It does not.** No model run in any condition used more than **one** search.

## Setup

Ten 2-hop questions in the shape real search traffic takes (`eval/searchqa/ceo_hops.json`):
company → current CEO → a stable fact about that CEO. Hop 1 changed *after* the model's
knowledge cutoff (Intel/Lip-Bu Tan 2025-03, Conagra/Brase 2026-06, IndiGo/Walsh 2026-08, …);
hop 2 is a pre-cutoff fact about a person the model has no reason to be thinking about. Every
fact verified against a primary or major-outlet source in July 2026.

Search tool is **live keenable web search**, cached per query string so repeats cost nothing and
every model sees byte-identical results. Model is `moe-kirk` (35B-A3B trained on the
Kirk/Japan/Iran corpus) — irrelevant to CEOs, so it stands in for "a model with stale knowledge
of this domain". 4 repeats × 10 questions.

## Results

| | told "skip what you know" | told "verify recent facts with search" |
|---|---|---|
| accuracy | 20/40 | 20/40 |
| mean searches | 0.55 | 0.53 |
| **max searches used by any run** | **1** | **1** |
| answered with 0 searches | 18/40 | 19/40 |
| answered with the *stale* entity | 19/40 | 18/40 |
| ever *searched* the stale entity | **0/40** | **0/40** |

Split by how much the model searched:

| | 0 searches | 1 search |
|---|---|---|
| told "skip what you know" | 0/18 correct (**0%**) | 20/22 (**91%**) |
| told "verify with search" | 0/19 correct (**0%**) | 20/21 (**95%**) |

## What this means

**1. There are no hops to save.** A modern web search API collapses a 2-hop entity-bridge
question into one query, because it returns *summarised snippets*, not raw pages. `current CEO
of Starbucks` comes back with "Brian Niccol … from Chipotle Mexican Grill" — both hops in one
result. The agent never needs a second search, so knowing hop 1 in advance saves nothing.

**2. The real pathology is the opposite of the one predicted.** Stale knowledge does not cause
*extra* searching; it causes *no* searching. 0 searches → 0% correct, every time. The model
answers `Pat Gelsinger`, `John Donahoe`, `Emma Walmsley` — its pre-cutoff world — without
checking. And it never once issues a query naming the outgoing CEO, i.e. it does not waste a hop
on a wrong premise, it simply skips the hop.

**3. Not an artifact of the prompt.** The obvious objection is that the default prompt says "if
you already know part of the answer, do not search for that part". Replacing that with "the
question may concern recent events, verify facts with SEARCH before answering" changed nothing:
20/40 either way, 18 vs 19 zero-search runs. The model does not believe it needs to check.

So the failure is **calibration, not knowledge** — the model does not know that it does not
know. Continued pretraining would make the un-searched answers right for topics we happen to
train on, and would do nothing for the ones we do not. That is not a better search agent, and
per our own results it would not even improve the underlying confidence: the same checkpoints
that learned the Kirk facts also assert Angela Merkel is dead with total confidence. Injecting
knowledge moves *which* facts are confidently asserted; it does not teach the model when to
doubt itself. The intervention this data argues for is knowing-when-to-search, which is the
subject of the agentic-RL "intrinsic knowledge boundary" line of work.

## Where the claim might still hold — not tested here

- **Vocabulary, not entities.** Anirudh's medical example is about not knowing what a *term
  means*, so you cannot form a good query at all. That is a different failure from entity lookup
  and this test does not cover it. It is the strongest remaining version of the claim.
- **Deeper chains.** 2 hops is shallow. 4–5 hop chains may not collapse into one query.
- **Weak or private corpora.** A live web index with snippet summarisation is the best possible
  case for the retriever. Over enterprise documents with no summaries, hops may survive.
- One model, ten questions, one seed. This is a vibe check that kills a hypothesis cheaply; it
  is not a measurement of anything.

## Files

`ceo_hops.json` — questions · `ceo-35B-{stale,neutral}.json` — runs · `websearch-cache.json` —
cached search results · `chains.json` — an earlier corpus-based attempt (BM25 over `data/real/`)
that collapsed to ~1 search for the same reason, which is what prompted using real web search
and real post-cutoff entities instead.

    python scripts/search_agent.py --tool web --chains eval/searchqa/ceo_hops.json \
        --model moe-kirk --repeats 4 --out eval/searchqa/ceo-35B-stale.json
