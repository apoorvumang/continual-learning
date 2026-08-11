"""Strip scraper boilerplate from the news corpus.

Audit of data/news2026/dsv4-janaug.jsonl (30,693 docs) found exactly one real defect class:

    381 docs   Reuters signup furniture that survived extraction, e.g.
               " Sign up [here.](undefined?location=article-paragraph&redirectUrl=%2Fworld%2F...)"

Two other patterns looked like defects under a naive regex and are NOT:

    7 docs     match /provided (context|summaries)/ -- but these are ordinary English about
               people: "Sangiorgi provided context on the drought in Somalia". Nothing leaked.
    540 docs   start with "Here is..." -- these are newsletter-format synthetic documents
               ("Dear Subscribers, Here is your daily summary"), a deliberate format from
               amplification, not an assistant preamble.

Deleting either would remove legitimate training text, so this only removes the first class, and
only the offending line rather than the document -- the surrounding article is fine and the fact
it reports is one we want learned.

    python scripts/clean_corpus.py --in data/news2026/dsv4-janaug.jsonl \
                                   --out data/news2026/dsv4-janaug-clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Anchored on the extraction artefact (`undefined?` / `redirectUrl=`), not on the words "sign up",
# so a genuine news sentence about signing up for something survives.
BOILERPLATE = re.compile(r"^.*(?:undefined\?location=|redirectUrl=%|\[here\.\]\(undefined).*$",
                         re.M)

# Documents that refer to the material they were generated from, or to the generation task itself.
# Whole documents are dropped, not lines: the reference is usually load-bearing in the sentence.
#
# Every pattern here was checked against samples first, because the obvious regexes are mostly
# WRONG on this corpus. Measured on 386k synthetic documents:
#
#   "the article"        1841 docs, and legitimate -- the DOC_TYPES list includes "a reader's
#                        letter to an editor responding to the coverage", for which referring to
#                        an article is the format, not a leak.
#   "not stated/included" 920 docs, and legitimate -- "the government has not specified a
#                        duration", "no details were provided on the intended targets" is how
#                        reporting hedges.
#   "as an AI"              8 docs, and legitimate -- "such as an AI giving relationship advice",
#                        "an AI center" hit by a naive \bas an ai\b.
#   "Where can I find updates?"  legitimate -- an FAQ document answering it.
#
# What is left is 0.19%: references to supplied material, and worse, the system prompt and the
# model's own planning leaking verbatim ("I will write 12 distinct documents", "Each document
# stands alone", "the user requested ~450 words"). That register is why DOC_TYPES carries a warning
# about fact-check formats: the model learned it once before and reproduced it in its reasoning
# when no context had been supplied at all.
META = re.compile(
    r"the (?:provided|supplied|source) (?:material|text|context|reports?|summar\w+|snippets?)"
    r"|(?:included|stated|mentioned|found) in the provided"
    r"|\bthe user (?:requested|asked)"
    r"|\bI will write \d+"
    r"|Each document (?:stands alone|must stand alone)"
    r"|uses only facts from"
    r"|\bthe available facts\b"
    r"|\bper the instructions\b", re.I)


def clean(text: str) -> tuple[str, int]:
    out, n = BOILERPLATE.subn("", text)
    if n:
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    docs_changed = lines_removed = dropped = total = 0
    with Path(args.out).open("w") as out:
        for line in Path(args.inp).open():
            if not line.strip():
                continue
            total += 1
            r = json.loads(line)
            if META.search(r["text"]):
                dropped += 1
                continue
            r["text"], n = clean(r["text"])
            if n:
                docs_changed += 1
                lines_removed += n
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{total} docs in, {total - dropped} out")
    print(f"  {dropped} dropped for referring to their source material or the generation task"
          f" ({dropped/max(total,1):.3%})")
    print(f"  {docs_changed} had {lines_removed} boilerplate lines stripped")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
