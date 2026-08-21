"""Collect the event classes the knowledge-cutoff benchmark actually scores.

That benchmark deliberately counts only low/medium-predictability events -- "assassinations, sudden
deaths, shock resignations, upsets" -- and excludes anything forecastable. It also ships
`control_alive` rows (living famous people) and `fake_event` rows to catch confabulation. So a corpus
optimised for token volume will be dominated by material the benchmark ignores, and a model trained
on it can still fail every question.

Wikipedia maintains exactly the right index pages, dated and structured:

  Deaths in <Month> <Year>   every notable death, with name, age, nationality, what they were known
                             for, and cause where reported. This is the single densest source of the
                             benchmark's primary event class, and it is precisely the knowledge that
                             produces "confidently wrong" answers when missing -- the model asserts
                             someone is alive.
  <Year> in <country>        national-level events, which catch resignations and political upsets
                             that the daily current-events pages summarise only briefly.

One document per person keeps each fact addressable, rather than burying a death in a list of four
hundred. The date is the death date, not the page date, so the curriculum places it correctly.

    python scripts/build_deaths_corpus.py --start 2025-04 --end 2026-08 --out data/news17/deaths
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
UA = "inception-research/1.0 (apoorv@inceptionlabs.ai)"
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# Entries look like:  *[[Name]], age, nationality, occupation, cause.<ref .../>
ENTRY = re.compile(r"^\*\s*\[\[([^\]|]+)(?:\|[^\]]*)?\]\]\s*,?\s*(.*)$")
DAY_HDR = re.compile(r"^===?\s*(\d{1,2})\s*===?\s*$")
CLEAN = [
    (re.compile(r"<ref[^>]*>.*?</ref>", re.S), ""),
    (re.compile(r"<ref[^>]*/>"), ""),
    (re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]"), r"\2"),
    (re.compile(r"\[\[([^\]]+)\]\]"), r"\1"),
    (re.compile(r"\{\{[^{}]*\}\}"), ""),
    (re.compile(r"'''?"), ""),
    (re.compile(r"\s+"), " "),
]


def clean(s: str) -> str:
    for pat, rep in CLEAN:
        s = pat.sub(rep, s)
    return s.strip(" ,.;")


def wikitext(page: str) -> str:
    u = (f"{API}?action=parse&page={urllib.parse.quote(page)}&prop=wikitext&format=json"
         "&formatversion=2")
    for a in range(5):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read())
            return (d.get("parse") or {}).get("wikitext") or ""
        except Exception:                                        # noqa: BLE001
            time.sleep(min(40, 3 * 2 ** a))
    return ""


def months(start: str, end: str):
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-04")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--out", default="data/news17/deaths")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grand = 0

    for y, m in months(args.start, args.end):
        label = f"{y:04d}-{m:02d}"
        mp = out / f"{label}.jsonl"
        if mp.exists() and mp.stat().st_size > 0:
            t = sum(json.loads(l).get("est_tokens", 0) for l in mp.open() if l.strip())
            print(f"[{label}] exists, {t/1e6:.3f}M tokens, skip", flush=True)
            grand += t
            continue
        wt = wikitext(f"Deaths in {MONTHS[m-1]} {y}")
        if not wt:
            print(f"[{label}] no wikitext", flush=True)
            continue
        rows, day = [], None
        for line in wt.split("\n"):
            h = DAY_HDR.match(line.strip())
            if h:
                day = int(h.group(1))
                continue
            mt = ENTRY.match(line.strip())
            if not mt or day is None:
                continue
            name = clean(mt.group(1))
            rest = clean(mt.group(2))
            if not name or len(rest) < 12:
                continue
            date = f"{y:04d}-{m:02d}-{day:02d}"
            # Written as prose so it reads like the rest of the corpus rather than a list row.
            body = (f"{name} died on {MONTHS[m-1]} {day}, {y}. {rest}."
                    if not rest.endswith(".") else
                    f"{name} died on {MONTHS[m-1]} {day}, {y}. {rest}")
            rows.append({"tier": "A", "date": date,
                         "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(name)}",
                         "domain": "en.wikipedia.org",
                         "title": f"Death of {name}", "text": body,
                         "est_tokens": max(1, len(body) // 4), "src": "wikipedia-deaths"})
        with mp.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        t = sum(r["est_tokens"] for r in rows)
        grand += t
        print(f"[{label}] {len(rows)} deaths, {t/1e3:.0f}k tokens", flush=True)
        time.sleep(0.5)

    print(f"DEATHS TOTAL ~{grand/1e6:.2f}M tokens")
    print("note: short documents by design -- this is high-value-per-token, not a volume source")


if __name__ == "__main__":
    main()
