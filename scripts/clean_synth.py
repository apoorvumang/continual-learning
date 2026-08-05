"""Audit and clean the synthetic corpus. Run this before every training run.

Three times now, text describing the *generation task* has leaked into the corpus and been learnt
as ordinary prose. Each was invisible in aggregate statistics and obvious on reading ten random
documents, so this tool does both: it drops known contamination, reports what it found as a
percentage, and prints samples to read.

  1. Leading format headers -- "**Document 3: Broadcast Script**", "**10. Market Note:**" -- from
     the prompt numbering the requested formats. 29% of documents. The model learnt to open every
     reply with a numbered header: instruction-following fell to 1/40.
  2. Source-referring language -- "the source material indicates", "not stated in the provided
     material". 1.6% of documents, concentrated in the fact-check format (27.5% of those, against
     <=2.6% elsewhere), because a fact-check is inherently *about a source*. The model reproduced
     that register with no context supplied, saying "the provided news snippets" when nothing had
     been provided. The format is gone from amplify_news.py; this drops the residue.
  3. (Not fixable here, for the record) using the tokenizer's EOS as a document separator, which
     for Qwen3.5 is the chat turn-end token. See train_sdf_lora.py.

There is no safe level of contamination -- 29% made the model do it constantly, 1.6% made it do
it occasionally. Treat any non-zero rate as a bug in the generator, not an acceptable residue.

    python scripts/clean_synth.py                      # audit + clean + samples
    python scripts/clean_synth.py --audit-only          # report without writing
    python scripts/clean_synth.py --samples 15          # read more documents
"""

from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
from pathlib import Path

# --- 1. leading scaffolding -------------------------------------------------------------------
# A first line naming one of the requested formats, optionally numbered and/or bolded. Anchored to
# the first line only: a numbered list *inside* a document is legitimate.
# The `#{1,6}` alternative matters: the generator emits markdown headings as well as bold, e.g.
# "### 6. Local Newspaper Coverage" and "### Document 1: Academic Note". An earlier version only
# matched `**` and bare prefixes, so 6,850 documents kept a visible "### Document 1:" line and were
# then thrown away by the contamination filter rather than simply cleaned. Found by reading four
# random samples, not by any aggregate number.
LEAD_HEADER = re.compile(
    r"^\s*(?:#{1,6}\s*)?\**\s*(?:(?:Document|Doc)\s*\d+\s*[:.\-]?|\d{1,2}\s*[.):])\s*"
    r"[^\n]{0,120}?\**\s*$", re.I)
# A markdown or bold heading that names one of the requested formats, with or without a number.
LEAD_TITLE = re.compile(r"^\s*(?:#{1,6}\s*[^\n]{3,120}|\*\*[^\n*]{3,120}\*\*)\s*$")
FORMAT_WORDS = re.compile(
    r"wire (?:service|brief)|news report|analysis|explainer|encyclopedia|timeline|faq|"
    r"question-and-answer|live ?blog|profile|newspaper|broadcast script|newsletter|"
    r"fact.?check|retrospective|briefing memo|editorial|academic note|market note|"
    r"transcript|letter to the editor|summary|follow-up report", re.I)

# --- 2. task-referring language --------------------------------------------------------------
# Deliberately specific. "supplied weapons" and "the ministry provided information" are ordinary
# news phrasing, so each pattern requires a word that can only mean *the source document*.
CONTAMINATION = {
    "source material": r"\bsource material\b",
    "provided/supplied material": r"\b(provided|supplied|given|attached)\s+(news\s+)?"
                                  r"(snippet|material|excerpt|passage)s?\b",
    "the provided/supplied text": r"\bthe (provided|supplied|given|above|following)\s+"
                                  r"(text|article|document|report|snippet)s?\b",
    "the text states/mentions": r"\bthe (text|passage|excerpt|material|source)\s+"
                                r"(states|mentions|says|indicates|notes|does not mention)\b",
    "research notes": r"\bresearch notes\b",
    "not mentioned in the source": r"\bnot (mentioned|stated|specified|provided) in the\s+"
                                   r"(text|article|material|source|snippet|document)s?\b",
    "according to the provided": r"\baccording to the (provided|supplied|above|following)\b",
    "based on the provided": r"\bbased on the (provided|supplied|above|following)\s+"
                             r"(text|material|article|information|snippet|document|report)s?\b",
    "residual Document N": r"\bDocument\s+\d+\s*[:.]",
    "fact-check scaffolding": r"\*\*(Claim|Verdict)\s*\d*\s*[:.]",
}
COMPILED = {k: re.compile(v, re.I) for k, v in CONTAMINATION.items()}


def strip_lead(text: str) -> tuple[str, bool]:
    """Peel at most two leading scaffolding lines (a numbered header, then a bold title)."""
    lines = text.split("\n")
    changed = False
    for _ in range(2):
        while lines and not lines[0].strip():
            lines.pop(0)
            changed = True
        if not lines:
            break
        first = lines[0]
        if LEAD_HEADER.match(first) or (LEAD_TITLE.match(first) and FORMAT_WORDS.search(first)):
            lines.pop(0)
            changed = True
            continue
        break
    return "\n".join(lines).strip(), changed


def contamination(text: str) -> list[str]:
    return [name for name, pat in COMPILED.items() if pat.search(text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="data/news2026/synth.jsonl")
    ap.add_argument("--out", dest="dst", default="data/news2026/synth-clean.jsonl")
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--samples", type=int, default=8,
                    help="random documents to print for reading; 0 to skip")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    counts = {k: 0 for k in CONTAMINATION}
    examples: dict[str, str] = {}
    n = kept = stripped = dropped_short = dropped_dirty = 0
    rng = random.Random(args.seed)
    reservoir: list[str] = []

    out = None if args.audit_only else open(args.dst, "w")
    for line in open(args.src):
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        text, changed = strip_lead(r["text"])
        stripped += changed

        bad = contamination(text)
        for name in bad:
            counts[name] += 1
            if name not in examples:
                m = COMPILED[name].search(text)
                examples[name] = " ".join(text[max(0, m.start() - 80):m.end() + 90].split())
        if bad:
            dropped_dirty += 1
            continue
        if len(text) < args.min_chars:
            dropped_short += 1
            continue

        kept += 1
        if out:
            r["text"] = text
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        # reservoir sample of what actually survives, for reading
        if args.samples:
            if len(reservoir) < args.samples:
                reservoir.append(text)
            elif rng.random() < args.samples / kept:
                reservoir[rng.randrange(args.samples)] = text
    if out:
        out.close()

    print(f"{n} documents in\n")
    print(f"  {stripped:7d} ({stripped/n:6.2%})  had leading format scaffolding stripped")
    print(f"  {dropped_dirty:7d} ({dropped_dirty/n:6.2%})  DROPPED for task-referring language")
    print(f"  {dropped_short:7d} ({dropped_short/n:6.2%})  dropped as too short after stripping")
    print(f"  {kept:7d} ({kept/n:6.2%})  kept\n")

    print("contamination by pattern:")
    for name in CONTAMINATION:
        c = counts[name]
        flag = "  <-- " if c else "      "
        print(f"  {c:7d} ({c/n:6.3%}){flag}{name}")
        if c and name in examples:
            print(f"            e.g. ...{examples[name][:120]}...")
    total = dropped_dirty
    print(f"\n{'CLEAN' if total == 0 else 'CONTAMINATED'}: {total} of {n} documents "
          f"({total/n:.2%}) referred to their own source material.")
    if total:
        print("Any non-zero rate is a generator bug, not an acceptable residue: 29% made the model\n"
              "do it constantly, 1.6% made it do it occasionally. Fix amplify_news.py, do not just\n"
              "filter -- a format that comments on a source should not be requested at all.")

    if reservoir:
        print(f"\n{'='*100}\nREAD THESE. Aggregate statistics missed all three contamination bugs;\n"
              f"each was obvious in ten random documents.\n{'='*100}")
        for i, t in enumerate(reservoir, 1):
            print(f"\n--- sample {i} ---")
            print(textwrap.fill(" ".join(t.split())[:700], 98,
                                initial_indent="  ", subsequent_indent="  "))
    if not args.audit_only:
        print(f"\n-> {args.dst}")


if __name__ == "__main__":
    main()
