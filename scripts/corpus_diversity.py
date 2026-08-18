"""Measure corpus diversity, because it is the thing Active Reading says actually matters.

Meta's Active Reading ablations found diversity — not answer coverage — predicts downstream recall:
synthetic QA had the HIGHEST coverage of target answers in their comparison and still underperformed,
while lower Self-BLEU (more varied surface form) tracked better results.

We optimised coverage on the first two tau2 corpora and never measured diversity at all: all 698
pages amplified at 92-98 documents each, every one of the 46 tools mentioned 83+ times. If the Active
Reading arm wins, this is the number that should have predicted it, and if it loses despite better
diversity that is worth knowing too.

Reports Self-BLEU (lower = more diverse), distinct-n over the corpus, and mean pairwise Jaccard on a
sample, since Self-BLEU alone is slow and noisy at this size.

    python scripts/corpus_diversity.py --a data/tau/kb-synth.jsonl --b data/tau/ar-docs.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

WORD = re.compile(r"[a-z0-9_]+")


def toks(t: str) -> list:
    return WORD.findall(t.lower())


def ngrams(ws: list, n: int) -> Counter:
    return Counter(tuple(ws[i:i + n]) for i in range(len(ws) - n + 1))


def self_bleu(docs: list, n: int = 4) -> float:
    """Mean modified n-gram precision of each document against one other, sampled.

    A full Self-BLEU is O(k^2) in documents; against a single random partner it is O(k) and the mean
    is a fine estimator for comparing two corpora at the same sample size.
    """
    if len(docs) < 2:
        return 0.0
    rng = random.Random(0)
    tot = 0.0
    for i, d in enumerate(docs):
        j = rng.randrange(len(docs) - 1)
        if j >= i:
            j += 1
        a, b = ngrams(toks(d), n), ngrams(toks(docs[j]), n)
        if not a:
            continue
        overlap = sum(min(c, b.get(g, 0)) for g, c in a.items())
        tot += overlap / sum(a.values())
    return tot / len(docs)


def distinct_n(docs: list, n: int) -> float:
    seen, total = set(), 0
    for d in docs:
        g = ngrams(toks(d), n)
        seen |= set(g)
        total += sum(g.values())
    return len(seen) / max(total, 1)


def jaccard(docs: list, pairs: int = 400) -> float:
    rng = random.Random(1)
    if len(docs) < 2:
        return 0.0
    tot = 0.0
    for _ in range(pairs):
        a, b = rng.sample(range(len(docs)), 2)
        sa, sb = set(toks(docs[a])), set(toks(docs[b]))
        tot += len(sa & sb) / max(len(sa | sb), 1)
    return tot / pairs


def load(p: str, n: int, seed: int = 0) -> list:
    rows = []
    for l in Path(p).open():
        if not l.strip():
            continue
        r = json.loads(l)
        # documents carry "text"; Q/A corpora carry q/a -- compare the generated surface either way
        rows.append(r.get("text") or f"{r.get('q','')} {r.get('a','')}".strip())
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--n", type=int, default=1500, help="documents sampled per corpus")
    args = ap.parse_args()

    A, B = load(args.a, args.n), load(args.b, args.n)
    print(f"\ndiversity, {len(A)} vs {len(B)} documents sampled\n")
    print(f"{'metric':26} {args.label_a:>18} {args.label_b:>18}  {'better':>8}")
    rows = [
        ("Self-BLEU-4 (lower=better)", self_bleu(A), self_bleu(B), "lower"),
        ("distinct-3 (higher=better)", distinct_n(A, 3), distinct_n(B, 3), "higher"),
        ("distinct-5 (higher=better)", distinct_n(A, 5), distinct_n(B, 5), "higher"),
        ("pairwise Jaccard (lower)", jaccard(A), jaccard(B), "lower"),
        ("mean words/doc", sum(len(toks(d)) for d in A) / len(A),
         sum(len(toks(d)) for d in B) / len(B), "-"),
    ]
    for name, va, vb, want in rows:
        if want == "lower":
            win = args.label_b if vb < va else args.label_a
        elif want == "higher":
            win = args.label_b if vb > va else args.label_a
        else:
            win = "-"
        print(f"{name:26} {va:18.4f} {vb:18.4f}  {win:>8}")
    print()


if __name__ == "__main__":
    main()
