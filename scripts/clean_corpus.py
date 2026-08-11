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

    docs_changed = lines_removed = 0
    rows = []
    for line in Path(args.inp).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["text"], n = clean(r["text"])
        if n:
            docs_changed += 1
            lines_removed += n
        rows.append(r)

    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} docs, {docs_changed} cleaned, {lines_removed} boilerplate lines removed")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
