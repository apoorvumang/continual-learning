"""Merge the collected streams into one dated, date-ordered continued-pretraining set.

Format, per the spec for this run:

    user       "The following is a document from 2025-04-17:\\n\\n"
    assistant  <the document>

written as a two-message record so `megatron pt --loss_scale default` supervises only the document.
That matters: `pt` defaults to `loss_scale: all`, and the label dump from our earlier runs shows only
the BOS masked -- every previous run trained the model to predict its own wrapper text too. `pt` also
sets `use_chat_template=False`, so no chat tokens are inserted and the base model stays in-domain.

Curriculum: records come out in strict date order, oldest first, so training walks April 2025 forward
to the present. The trainer must also be told not to shuffle, or this ordering is discarded.

Four streams merge here, and they are deliberately unequal in character:

  wiki    Wikipedia current events -- human-curated "what happened" plus the cited article text.
          The citations are Reuters, AP, Guardian, BBC, Al Jazeera, CNBC, which CC-NEWS cannot reach.
  hn      HackerNews above a point threshold, fetched from the publisher directly. A salience filter.
  deaths  one document per notable death. Tiny in tokens, but it is the event class the
          knowledge-cutoff benchmark actually scores, and the class whose absence produces confident
          errors rather than honest ignorance.
  docs    CC-NEWS. Volume and long-tail coverage, already filtered for language, syndication,
          content farms and per-domain domination.

Deduplication runs across streams as well as within them: an article can arrive via HN and via
CC-NEWS, and the same story is syndicated under many URLs.

    python scripts/build_cpt_dataset.py --target-tokens 400e6 --out data/news17/cpt.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

WS = re.compile(r"\s+")
PREFIX = "The following is a document from {date}:\n\n"

# Relative weight when subsampling to a token budget. Curated and salience-selected material is kept
# preferentially over bulk crawl; deaths are never dropped.
STREAM_PRIORITY = {"deaths": 0, "wiki": 1, "hn": 2, "docs": 3}


def norm_key(text: str) -> str:
    return hashlib.blake2b(WS.sub(" ", text[:400].lower()).strip().encode(),
                           digest_size=16).hexdigest()


def load_stream(root: Path, name: str) -> list:
    d = root / name
    if not d.exists():
        return []
    rows = []
    files = sorted(d.glob("*.jsonl")) if d.is_dir() else [d]
    for p in files:
        for line in p.open():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:                                    # noqa: BLE001
                continue
            txt = r.get("text") or ""
            date = (r.get("date") or r.get("published_at") or "")[:10]
            if not txt or not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
                continue
            rows.append({"date": date, "text": txt, "url": r.get("url") or "",
                         "domain": r.get("domain") or "", "stream": name,
                         "est_tokens": r.get("est_tokens") or max(1, len(txt) // 4)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/news17")
    ap.add_argument("--streams", default="deaths,wiki,hn,docs")
    ap.add_argument("--out", default="data/news17/cpt.jsonl")
    ap.add_argument("--target-tokens", type=float, default=0,
                    help="0 keeps everything; otherwise drop from the lowest-priority stream first")
    ap.add_argument("--min-chars", type=int, default=350)
    ap.add_argument("--max-chars", type=int, default=24000)
    ap.add_argument("--report", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    all_rows, per = [], collections.Counter()
    for name in [s.strip() for s in args.streams.split(",") if s.strip()]:
        rows = load_stream(root, name)
        per[name] = len(rows)
        all_rows += rows
        print(f"  {name:8} {len(rows):8d} docs  "
              f"{sum(r['est_tokens'] for r in rows)/1e6:7.1f}M tokens", flush=True)

    # Cross-stream dedup. Prefer the higher-priority stream when the same text arrives twice.
    all_rows.sort(key=lambda r: STREAM_PRIORITY.get(r["stream"], 9))
    seen_txt, seen_url, kept = set(), set(), []
    dropped = collections.Counter()
    # Death entries are one sentence by design ("X died on April 1, 2025. 79, Mexican actor.") and
    # a 350-character floor discards every one of them -- 14,044 documents, and the densest
    # benchmark-relevant material in the corpus. Packing makes short records free, so exempt them.
    SHORT_OK = {"deaths"}
    for r in all_rows:
        lo = 40 if r["stream"] in SHORT_OK else args.min_chars
        if not (lo <= len(r["text"]) <= args.max_chars):
            dropped["length"] += 1
            continue
        k = norm_key(r["text"])
        if k in seen_txt:
            dropped["dup_text"] += 1
            continue
        u = r["url"].split("?")[0].rstrip("/")
        if u and u in seen_url:
            dropped["dup_url"] += 1
            continue
        seen_txt.add(k)
        if u:
            seen_url.add(u)
        kept.append(r)
    print(f"\nafter dedup/length: {len(kept)} docs, "
          f"{sum(r['est_tokens'] for r in kept)/1e6:.1f}M tokens  dropped={dict(dropped)}")

    # Subsample to budget by dropping the lowest-priority stream first, uniformly within it so the
    # month distribution is preserved and the curriculum stays even.
    if args.target_tokens:
        tot = sum(r["est_tokens"] for r in kept)
        if tot > args.target_tokens:
            rng = random.Random(args.seed)
            by_stream = collections.defaultdict(list)
            for r in kept:
                by_stream[r["stream"]].append(r)
            order = [s for s in sorted(by_stream, key=lambda s: -STREAM_PRIORITY.get(s, 9))
                     if s != "deaths"]
            excess = tot - args.target_tokens
            for s in order:
                if excess <= 0:
                    break
                pool = by_stream[s]
                rng.shuffle(pool)
                drop_tok, keep_from = 0, len(pool)
                for i, r in enumerate(pool):
                    if drop_tok >= excess:
                        keep_from = i
                        break
                    drop_tok += r["est_tokens"]
                else:
                    keep_from = len(pool)
                by_stream[s] = pool[keep_from:]
                excess -= drop_tok
                print(f"  trimmed {s}: dropped {drop_tok/1e6:.1f}M tokens", flush=True)
            kept = [r for s in by_stream for r in by_stream[s]]

    # Curriculum order. Ties inside a day are shuffled so streams interleave rather than arriving in
    # blocks, which would make the within-day distribution lumpy.
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    kept.sort(key=lambda r: r["date"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    months = collections.Counter()
    streams = collections.Counter()
    tok = 0
    with out.open("w") as f:
        for r in kept:
            rec = {"messages": [
                {"role": "user", "content": PREFIX.format(date=r["date"])},
                {"role": "assistant", "content": r["text"]},
            ]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            months[r["date"][:7]] += r["est_tokens"]
            streams[r["stream"]] += r["est_tokens"]
            tok += r["est_tokens"]

    print(f"\nwrote {len(kept)} records, ~{tok/1e6:.1f}M tokens -> {out}")
    print("\nby month (curriculum order):")
    for m in sorted(months):
        print(f"  {m}  {months[m]/1e6:6.1f}M")
    print("\nby stream:")
    for s, v in streams.most_common():
        print(f"  {s:8} {v/1e6:7.1f}M  ({v/tok:5.1%})")
    if args.report:
        Path(args.report).write_text(json.dumps(
            {"docs": len(kept), "tokens": tok,
             "by_month": dict(months), "by_stream": dict(streams)}, indent=1))


if __name__ == "__main__":
    main()
