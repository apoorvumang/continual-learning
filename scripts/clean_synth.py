"""Strip generator scaffolding from the synthetic corpus.

Arm S trained on the raw output and the merged model was destroyed: instruction compliance
1/40, and asked for the capital of France it replied "**10. Market Note: Financial Implications
of the 202...". The cause was in the data, not the training. 29.3% of documents opened with an
instruction artifact:

    20.5%  **Document 3: Broadcast Script...**
     8.8%  **10. Market Note: Financial Implications...**

The prompt asked for {n} documents and numbered the requested formats 1..12, so the generator
echoed those numbers as headings. `parse_docs` stripped a bare "Document N" prefix but not the
markdown-bolded or numbered variants, so nearly a third of 102M tokens taught the model to open
every reply with a numbered format header -- which is precisely the observed failure.

Run this over synth.jsonl before training. It rewrites in place-ish (new file), reports what it
changed, and drops documents that are nothing but scaffolding.

    python scripts/clean_synth.py --in data/news2026/synth.jsonl \
        --out data/news2026/synth-clean.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# A leading heading line that names one of the requested formats, optionally numbered and/or
# bolded. Anchored to the first line only: a numbered list *inside* a document is legitimate.
LEAD_HEADER = re.compile(
    r"^\s*\**\s*(?:(?:Document|Doc)\s*\d+\s*[:.\-]?|\d{1,2}\s*[.):])\s*[^\n]{0,120}?\**\s*$",
    re.I)
# Bare bolded title line with no number, e.g. "**Market Note: Financial Implications**"
LEAD_TITLE = re.compile(r"^\s*\*\*[^\n*]{3,120}\*\*\s*$")
FORMAT_WORDS = re.compile(
    r"wire (?:service|brief)|news report|analysis|explainer|encyclopedia|timeline|faq|"
    r"question-and-answer|live ?blog|profile|newspaper|broadcast script|newsletter|"
    r"fact.?check|retrospective|briefing memo|editorial|academic note|market note|"
    r"transcript|letter to the editor|summary|follow-up report", re.I)


def clean(text: str) -> tuple[str, bool]:
    lines = text.split("\n")
    changed = False
    # Peel at most two leading scaffolding lines (a numbered header then a bold title).
    for _ in range(2):
        while lines and not lines[0].strip():
            lines.pop(0)
            changed = True
        if not lines:
            break
        first = lines[0]
        is_numbered = bool(LEAD_HEADER.match(first))
        is_title = bool(LEAD_TITLE.match(first)) and bool(FORMAT_WORDS.search(first))
        if is_numbered or is_title:
            lines.pop(0)
            changed = True
            continue
        break
    out = "\n".join(lines).strip()
    return out, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="data/news2026/synth.jsonl")
    ap.add_argument("--out", dest="dst", default="data/news2026/synth-clean.jsonl")
    ap.add_argument("--min-chars", type=int, default=300)
    args = ap.parse_args()

    n = kept = touched = dropped = 0
    with open(args.src) as fin, open(args.dst, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            text, changed = clean(r["text"])
            touched += changed
            if len(text) < args.min_chars:
                dropped += 1
                continue
            r["text"] = text
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1
    print(f"{n} documents in, {kept} kept, {dropped} dropped as too short after cleaning")
    print(f"{touched} had leading scaffolding removed ({touched/max(1,n):.1%})")
    print(f"-> {Path(args.dst)}")


if __name__ == "__main__":
    main()
