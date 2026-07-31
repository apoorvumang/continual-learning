"""Build a broad 2026 news corpus for continued pretraining, day by day.

Why not just query a search API for "top news" per day: keenable is a search engine, not a news
feed, so generic queries return site homepages, and topical queries return 1% major-outlet
results against 99% SEO filler. Measured, not assumed.

Instead this seeds from Wikipedia's Current Events portal, which has one curated page per day
listing what happened, grouped by category, with **an inline citation to the source article for
almost every item**. That gives three things for free: complete day coverage, human-curated
selection of what mattered, and a list of real article URLs whose domains are Reuters, AP,
Guardian, BBC, Al Jazeera, CNBC rather than content farms.

Two document types come out, both kept:
  summary  the day's events as prose, stripped of wiki markup. Dense, dated, no boilerplate.
  article  full text of a cited source, fetched through keenable.

Every document carries its date, so a temporal train/test split is possible later -- train on
Jan-May and hold out Jun-Jul, which is what a real claim about moving the knowledge cutoff
needs, as opposed to the same-topics-as-eval design of the SDF runs.

Resumable: rerun after an interruption and it skips days and URLs already collected.

    python scripts/retrieve_news_2026.py --start 2026-01-01 --end 2026-07-31
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from keenable import fetch  # noqa: E402
from retrieve_docs import BLOCKED_SUBSTRINGS, domain_of  # noqa: E402

WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "continual-learning-research/1.0 (apoorv@inceptionlabs.ai)"}
MIN_ARTICLE_CHARS = 500
_wiki_lock = threading.Lock()
_wiki_next = [0.0]


def wiki_wikitext(title: str) -> str:
    """One request per day of the corpus, so politeness matters more than speed: serialise to
    ~2/s and identify ourselves, per Wikimedia's API etiquette."""
    with _wiki_lock:
        gap = _wiki_next[0] - time.monotonic()
        _wiki_next[0] = max(time.monotonic(), _wiki_next[0]) + 0.5
    if gap > 0:
        time.sleep(gap)
    url = (f"{WIKI_API}?action=parse&prop=wikitext&format=json"
           f"&page={urllib.parse.quote(title)}")
    for attempt in range(4):
        try:
            d = json.load(urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=60))
            return "" if "error" in d else d["parse"]["wikitext"]["*"]
        except Exception:
            if attempt == 3:
                return ""
            time.sleep(2 * (attempt + 1))
    return ""


# ---------------------------------------------------------------- wiki markup
def strip_markup(w: str) -> str:
    """Wiki markup -> plain prose. Deliberately conservative: anything ambiguous is dropped
    rather than half-rendered, since this text is training data and mangled markup would be
    learned as text."""
    w = re.sub(r"^\{\{Current events[^\n]*\n", "", w)
    w = re.sub(r"<!--.*?-->", "", w, flags=re.S)
    w = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", w)   # [url (Reuters)] -> (Reuters)
    w = re.sub(r"\{\{ship\|[^}]*?\|([^|}]+)\|[^}]*\}\}", r"\1", w)
    w = re.sub(r"\{\{[^{}]*\}\}", "", w)
    w = re.sub(r"\{\{[^{}]*\}\}", "", w)                        # nested templates
    w = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", w)          # [[target|shown]] -> shown
    w = re.sub(r"\[\[([^\]]*)\]\]", r"\1", w)
    w = re.sub(r"'''?", "", w)
    w = re.sub(r"</?[a-zA-Z][^>]*>", "", w)
    lines = []
    for ln in w.splitlines():
        ln = re.sub(r"^\*+\s*", "", ln).strip()
        if ln and ln != "}}":
            lines.append(ln)
    return "\n".join(lines)


def citations(w: str) -> list[str]:
    urls = re.findall(r"\[(https?://[^\s\]]+)", w)
    out, seen = [], set()
    for u in urls:
        u = u.rstrip(".,);")
        d = domain_of(u)
        if u in seen or any(b in d for b in BLOCKED_SUBSTRINGS):
            continue
        seen.add(u)
        out.append(u)
    return out


def iso_date(v, fallback: str) -> str:
    """keenable returns published_at as a unix timestamp for some sources and ISO for others.
    Normalise, because the whole point of the date field is a temporal train/test split."""
    if v is None or v == "":
        return fallback
    try:
        n = float(v)
        if 9.4e8 < n < 2.2e9:                     # plausible seconds since epoch
            return dt.datetime.utcfromtimestamp(n).date().isoformat()
    except (TypeError, ValueError):
        pass
    m = re.match(r"(\d{4}-\d{2}-\d{2})", str(v))
    return m.group(1) if m else fallback


def daterange(start: str, end: str):
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    while d0 <= d1:
        yield d0
        d0 += dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--out", default="data/news2026")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summaries-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs_path, seen_path = out / "docs.jsonl", out / "seen_urls.txt"

    seen: set[str] = set()
    have_days: set[str] = set()
    if docs_path.exists():
        for line in docs_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["kind"] == "summary":
                    have_days.add(r["date"])
                else:
                    seen.add(r["url"])
    if seen_path.exists():                       # urls tried and rejected, do not retry
        seen |= set(seen_path.read_text().split())
    print(f"resume: {len(have_days)} days of summaries, {len(seen)} urls already handled")

    fout = docs_path.open("a")
    fseen = seen_path.open("a")
    lock = threading.Lock()
    n_art = n_fail = 0

    def grab(job):
        date, url = job
        try:
            page = fetch(url)
        except Exception as e:
            return date, url, None, str(e)[:120]
        return date, url, page, None

    for day in daterange(args.start, args.end):
        iso = day.isoformat()
        title = f"Portal:Current events/{day.strftime('%Y %B')} {day.day}"
        if iso in have_days:
            pending = []
        else:
            w = wiki_wikitext(title)
            if not w:
                print(f"[{iso}] no wiki page", flush=True)
                continue
            summary = strip_markup(w)
            urls = citations(w)
            with lock:
                fout.write(json.dumps({"kind": "summary", "date": iso, "url": title,
                                       "domain": "en.wikipedia.org",
                                       "title": f"News summary for {iso}",
                                       "published_at": iso, "text": summary},
                                      ensure_ascii=False) + "\n")
                fout.flush()
            pending = [(iso, u) for u in urls if u not in seen]
            seen.update(u for _, u in pending)
            print(f"[{iso}] summary {len(summary):5d} chars, {len(urls):3d} citations, "
                  f"{len(pending):3d} new urls", flush=True)

        if args.summaries_only or not pending:
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for date, url, page, err in pool.map(grab, pending):
                content = (page or {}).get("content") or ""
                if err or len(content) < MIN_ARTICLE_CHARS:
                    n_fail += 1
                    fseen.write(url + "\n")
                    continue
                n_art += 1
                fout.write(json.dumps(
                    {"kind": "article", "date": date, "url": url, "domain": domain_of(url),
                     "title": (page.get("title") or "")[:300],
                     "published_at": iso_date(page.get("published_at"), date),
                     "text": content}, ensure_ascii=False) + "\n")
        fout.flush()
        fseen.flush()
        print(f"    -> cumulative {n_art} articles, {n_fail} unusable", flush=True)

    fout.close()
    fseen.close()
    print(f"done: {n_art} articles fetched, {n_fail} unusable -> {docs_path}")


if __name__ == "__main__":
    main()
