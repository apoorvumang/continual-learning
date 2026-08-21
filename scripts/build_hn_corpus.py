"""Collect what the tech community judged important: HackerNews-ranked stories, fetched directly.

The point of this collector is selection, not volume. arXiv gives thousands of papers a month where
a handful matter; HuggingFace listings are metadata; release notes are changelogs. HN carries an
explicit human salience signal -- points -- and the Algolia API lets us filter on it, so we can ask
for "what mattered" rather than "what was published".

It also solves the gap in the news spine. CC-NEWS contains none of the major Western outlets, since
Reuters, AP, BBC, NYT, Guardian, TechCrunch and The Verge all block Common Crawl. HN links straight
at them, and fetching those URLs directly works where the crawl does not.

Measured volume for April 2025: 4,399 stories above 10 points, 2,885 above 20, 1,983 above 40, 1,284
above 80. At 20 points that is roughly 49k stories over seventeen months.

Every document keeps its date, so it merges with the news corpus and feeds the month-by-month
curriculum.

    python scripts/build_hn_corpus.py --start 2025-04 --end 2026-08 --min-points 20 \
        --workers 16 --out data/news17/hn
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = ("Mozilla/5.0 (compatible; inception-research/1.0; +mailto:apoorv@inceptionlabs.ai) "
      "corpus-collection")
ALGOLIA = "https://hn.algolia.com/api/v1/search_by_date"
WS = re.compile(r"\s+")

# Link targets that are not articles: code hosts, video, social, aggregators.
SKIP_URL = re.compile(
    r"(youtube\.com|youtu\.be|twitter\.com|x\.com/|reddit\.com|news\.ycombinator|"
    r"\.pdf$|\.zip$|\.mp4$|github\.com/[^/]+/[^/]+/?$|gitlab\.com|"
    r"linkedin\.com|facebook\.com|instagram\.com|tiktok\.com|"
    r"docs\.google\.com|drive\.google\.com|imgur\.com)", re.I)


def month_bounds(start: str, end: str):
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    while (y, m) <= (ey, em):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        lo = int(dt.datetime(y, m, 1).timestamp())
        hi = int(dt.datetime(ny, nm, 1).timestamp())
        yield f"{y:04d}-{m:02d}", lo, hi
        y, m = ny, nm


def api(url: str, tries: int = 5):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read())
        except Exception:                                        # noqa: BLE001
            time.sleep(min(45, 3 * 2 ** a))
    return None


def stories(lo: int, hi: int, min_points: int, window_days: int = 7) -> list:
    """All stories in the window above the threshold.

    Algolia caps a single query at 1000 hits regardless of paging, and a month above 20 points has
    roughly 2,900 -- so querying by month silently returned a third of what exists. Split the month
    into sub-windows small enough to stay under the cap.
    """
    step = window_days * 86400
    out = []
    a = lo
    while a < hi:
        b = min(a + step, hi)
        out += _stories_window(a, b, min_points)
        a = b
    return out


def _stories_window(lo: int, hi: int, min_points: int) -> list:
    out, page = [], 0
    while page < 60:
        u = (f"{ALGOLIA}?tags=story&numericFilters=created_at_i>{lo},created_at_i<{hi},"
             f"points>{min_points}&hitsPerPage=1000&page={page}")
        d = api(u)
        if not d or not d.get("hits"):
            break
        out += d["hits"]
        if page + 1 >= d.get("nbPages", 1):
            break
        page += 1
        time.sleep(0.3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-04")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--min-points", type=int, default=20)
    ap.add_argument("--out", default="data/news17/hn")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--window-days", type=int, default=7,
                    help="sub-window size; Algolia caps one query at 1000 hits")
    args = ap.parse_args()

    import trafilatura
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def fetch_one(h):
        url = h.get("url") or ""
        if not url or SKIP_URL.search(url):
            return None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                html = r.read(4 << 20)
        except Exception:                                        # noqa: BLE001
            return None
        try:
            txt = trafilatura.extract(html, include_comments=False, include_tables=False,
                                      fast=True, favor_precision=True)
        except Exception:                                        # noqa: BLE001
            return None
        if not txt or len(txt) < args.min_chars:
            return None
        title = WS.sub(" ", h.get("title") or "").strip()
        date = (h.get("created_at") or "")[:10]
        body = f"{title}\n\n{txt}"
        return {"tier": "A", "date": date, "url": url,
                "domain": url.split("/")[2] if "://" in url else "",
                "title": title, "text": body, "est_tokens": len(body) // 4,
                "src": "hn", "points": h.get("points"),
                "hn_url": f"https://news.ycombinator.com/item?id={h.get('objectID')}"}

    grand = 0
    for label, lo, hi in month_bounds(args.start, args.end):
        mp = out / f"{label}.jsonl"
        if mp.exists() and mp.stat().st_size > 0:
            t = sum(json.loads(l).get("est_tokens", 0) for l in mp.open() if l.strip())
            print(f"[{label}] exists, {t/1e6:.2f}M tokens, skip", flush=True)
            grand += t
            continue
        hits = stories(lo, hi, args.min_points, args.window_days)
        # sub-windows overlap at boundaries
        hits = list({h.get("objectID"): h for h in hits}.values())
        print(f"[{label}] {len(hits)} stories above {args.min_points} points, fetching...",
              flush=True)
        rows, t0 = [], time.time()
        with ThreadPoolExecutor(args.workers) as ex:
            for r in ex.map(fetch_one, hits):
                if r:
                    rows.append(r)
        # Same story often submitted twice; keep the higher-scoring copy.
        best = {}
        for r in rows:
            k = r["url"]
            if k not in best or (r.get("points") or 0) > (best[k].get("points") or 0):
                best[k] = r
        rows = sorted(best.values(), key=lambda r: r["date"])
        with mp.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        t = sum(r["est_tokens"] for r in rows)
        grand += t
        print(f"[{label}] kept {len(rows)}/{len(hits)} ({len(rows)/max(len(hits),1):.0%} fetchable), "
              f"{t/1e6:.2f}M tokens, {(time.time()-t0)/60:.1f}m", flush=True)
    print(f"HN TOTAL ~{grand/1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
