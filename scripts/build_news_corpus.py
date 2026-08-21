"""Collect a dated document corpus from CC-NEWS, April 2025 to the present.

Every document carries the date it was crawled, because the training format is
`The following is a document from <date>:` and the curriculum runs month by month. The date is part
of the objective, not metadata.

Two things about performance, both learned the hard way:

  parse in PROCESSES, not threads. trafilatura and py3langid are pure Python and GIL-bound, so a
  24-thread pool delivered roughly one core: the first attempt took four hours for a single month,
  which would have been sixty-eight hours for seventeen. Downloading is I/O and stays on threads.

  fetch at most a handful at a time. Common Crawl answers parallel downloads with 503; twenty-four
  concurrent fetches returned nothing else.

So each round downloads a batch with a small thread pool, then parses that batch across many
processes, then repeats until the month's token quota is met.

Filters, in rough order of how much they remove: English only (CC-NEWS is majority non-English --
the first sample was Iranian, Kazakh, Russian and Vietnamese by volume), >= 400 characters of
extracted text, a syndication-resistant duplicate key, and a block list for wire reprints and SEO
farms.

Deliberately NOT here: GDELT salience ranking. That needs a per-day entity join and can be applied
over this output later without re-downloading. This pass takes an even monthly sample so the
curriculum has uniform density.

    python scripts/build_news_corpus.py --start 2025-04 --end 2026-08 \
        --tokens-per-month 40e6 --parse-workers 28 --out data/news17/docs
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

CC = "https://data.commoncrawl.org/"
UA = "inception-research/1.0 (apoorv@inceptionlabs.ai; continual-pretraining corpus)"

# Raw CC-NEWS is a firehose, and a month of it is dominated by things nobody would call news. In the
# first collected month the MarketBeat syndication network alone (themarketsdaily, tickerreport,
# americanbankingnews, wkrb13, etfdailynews, theenterpriseleader) was 7.5% of documents -- thousands
# of auto-generated "NFJ Investment Group LLC Trims Stock Position in ..." filings, each naming a
# different company so no dedup key catches them. Aggregators and content farms took much of the rest.
DOMAIN_BLOCK = re.compile(
    r"(coinspeaker|cryptopolitan|benzinga\.com/pressreleases|globenewswire|prnewswire|"
    r"businesswire|einpresswire|openpr|newswire\.ca|finanzen\.net|finanznachrichten|"
    r"marketscreener|simplywall\.st|zacks\.com|stocktitan|accesswire|newsfilecorp|"
    # MarketBeat-syndicated 13F/analyst-note spam
    r"themarketsdaily|tickerreport|americanbankingnews|wkrb13|etfdailynews|theenterpriseleader|"
    r"modernreaders|dispatchtribunal|thelincolnianonline|macroaxis|marketbeat|"
    # aggregators and reprint mills
    r"menafn|devdiscourse|latestly|webindia123|indiaeducationdiary|openpr|"
    # exchange and token promo
    r"mexc\.com|bitget|gate\.io|binance\.com/en/news|coinmarketcap\.com/community)", re.I)

# Auto-generated filing and analyst-note headlines, which survive every other filter.
TITLE_BLOCK = re.compile(
    r"(Trims? Stock|Stock Position|Shares Sold by|Purchases New Shares|Sells \d|Buys \d|"
    r"Position (Raised|Lowered|Boosted|Trimmed|Increased|Decreased)|"
    r"Acquires New (Shares|Position|Stake)|13F|Short Interest (Up|Down)|"
    r"Price Target (Raised|Lowered)|Reiterates|Given a \"|Coverage Initiated|"
    r"Trading (Up|Down) [\d.]+%|Sets New \d+-(Week|Month|Year))", re.I)

# Domains whose output is worth keeping without a cap. Everything else is capped per month so no
# single site can dominate the curriculum, which also buys diversity for free.
TIER_A = re.compile(
    r"(reuters\.com|apnews\.com|bbc\.co|bbc\.com|nytimes\.com|washingtonpost\.com|wsj\.com|"
    r"ft\.com|bloomberg\.com|economist\.com|theguardian\.com|npr\.org|cnn\.com|cnbc\.com|"
    r"aljazeera\.com|dw\.com|france24\.com|politico\.|axios\.com|thehill\.com|"
    r"nature\.com|science\.org|newscientist\.com|scientificamerican\.com|"
    r"techcrunch\.com|theverge\.com|arstechnica\.com|wired\.com|zdnet\.com|theregister\.com|"
    r"venturebeat\.com|semianalysis\.com|ieee\.org|"
    r"nasdaq\.com|marketwatch\.com|barrons\.com|fortune\.com|businessinsider\.com|"
    r"forbes\.com|hbr\.org|"
    r"thehindu\.com|indianexpress\.com|hindustantimes\.com|economictimes\.indiatimes\.com|"
    r"scmp\.com|japantimes\.co\.jp|straitstimes\.com|smh\.com\.au|abc\.net\.au|"
    r"cbc\.ca|globeandmail\.com|irishtimes\.com|elpais\.com/english|spiegel\.de/international)",
    re.I)

WS = re.compile(r"\s+")
MIN_CHARS = 400


def norm_key(title: str, text: str) -> str:
    t = WS.sub(" ", (title or "").lower()).strip()
    b = WS.sub(" ", (text or "")[:300].lower()).strip()
    return hashlib.blake2b((t + "|" + b).encode(), digest_size=16).hexdigest()


def parse_warc(local_path: str) -> list:
    """Runs in a worker PROCESS. Extract article text from one WARC, then delete it."""
    from warcio.archiveiterator import ArchiveIterator
    import py3langid
    import trafilatura
    rows = []
    try:
        with open(local_path, "rb") as f:
            for rec in ArchiveIterator(f):
                if rec.rec_type != "response":
                    continue
                url = rec.rec_headers.get_header("WARC-Target-URI") or ""
                if DOMAIN_BLOCK.search(url):
                    continue
                date = (rec.rec_headers.get_header("WARC-Date") or "")[:10]
                try:
                    html = rec.content_stream().read()
                except Exception:                                # noqa: BLE001
                    continue
                txt = trafilatura.extract(html, include_comments=False, include_tables=False,
                                          fast=True, favor_precision=True)
                if not txt or len(txt) < MIN_CHARS:
                    continue
                if py3langid.classify(txt[:1500])[0] != "en":
                    continue
                title = ""
                try:
                    meta = trafilatura.extract_metadata(html)
                    title = (getattr(meta, "title", None) or "") if meta else ""
                except Exception:                                # noqa: BLE001
                    pass
                if TITLE_BLOCK.search(title):
                    continue
                rows.append({"tier": "A" if TIER_A.search(url) else "B",
                             "date": date, "url": url,
                             "domain": url.split("/")[2] if "://" in url else "",
                             "title": title, "text": txt,
                             "est_tokens": len(txt) // 4, "src": "cc-news",
                             "_k": norm_key(title, txt)})
    except Exception as e:                                       # noqa: BLE001
        return [{"_error": f"{os.path.basename(local_path)}: {type(e).__name__} {str(e)[:80]}"}]
    finally:
        try:
            os.unlink(local_path)
        except OSError:
            pass
    return rows


def fetch(path: str, tmp: str) -> str | None:
    local = os.path.join(tmp, os.path.basename(path))
    for attempt in range(6):
        try:
            req = urllib.request.Request(CC + path, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=900) as r, open(local, "wb") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
            return local
        except Exception:                                        # noqa: BLE001
            try:
                os.unlink(local)
            except OSError:
                pass
            if attempt < 5:
                time.sleep(min(120, 8 * 2 ** attempt) * (0.5 + random.random()))
    return None


def warc_paths(month: str) -> list:
    url = f"{CC}crawl-data/CC-NEWS/{month.replace('-', '/')}/warc.paths.gz"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return gzip.decompress(r.read()).decode().split()


def months(start: str, end: str) -> list:
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-04")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--tokens-per-month", type=float, default=40e6)
    ap.add_argument("--out", default="data/news17/docs")
    ap.add_argument("--parse-workers", type=int, default=28)
    ap.add_argument("--download-workers", type=int, default=5)
    ap.add_argument("--batch", type=int, default=14, help="WARCs fetched+parsed per round")
    ap.add_argument("--tier-b-cap", type=int, default=250,
                    help="max documents per month from any single non-tier-A domain")
    ap.add_argument("--tmp", default="/tmp/ccnews")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    ms = months(args.start, args.end)
    print(f"{len(ms)} months: {ms[0]} .. {ms[-1]}, target {args.tokens_per_month/1e6:.0f}M each",
          flush=True)

    pool = ProcessPoolExecutor(args.parse_workers)
    for month in ms:
        mp = out_dir / f"{month}.jsonl"
        tok = 0
        seen = set()
        if mp.exists():
            for line in mp.open():
                if line.strip():
                    r = json.loads(line)
                    tok += r.get("est_tokens", 0)
                    seen.add(r.get("_k", ""))
            if tok >= args.tokens_per_month:
                print(f"[{month}] already {tok/1e6:.1f}M tokens, skipping", flush=True)
                continue
        try:
            paths = warc_paths(month)
        except Exception as e:                                   # noqa: BLE001
            print(f"[{month}] path list failed: {str(e)[:90]}", flush=True)
            continue
        random.Random(args.seed).shuffle(paths)
        fout = mp.open("a")
        t0 = time.time()
        docs = 0
        i = 0
        per_dom = collections.Counter()
        capped = 0
        while tok < args.tokens_per_month and i < len(paths):
            batch = paths[i:i + args.batch]
            i += args.batch
            with ThreadPoolExecutor(args.download_workers) as dex:
                locals_ = [p for p in dex.map(lambda p: fetch(p, args.tmp), batch) if p]
            if not locals_:
                print(f"[{month}] whole batch failed to download", flush=True)
                continue
            for rows in pool.map(parse_warc, locals_):
                for r in rows:
                    if "_error" in r:
                        print(f"    {r['_error']}", flush=True)
                        continue
                    k = r.pop("_k")
                    if k in seen:
                        continue
                    if r["tier"] == "B":
                        if per_dom[r["domain"]] >= args.tier_b_cap:
                            capped += 1
                            continue
                        per_dom[r["domain"]] += 1
                    seen.add(k)
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                    tok += r["est_tokens"]
                    docs += 1
            fout.flush()
            el = time.time() - t0
            print(f"[{month}] {i:3d} warcs  {docs:7d} docs  {tok/1e6:6.2f}M tok  "
                  f"{el/60:5.1f}m  ({tok/1e6/max(el/60,.01):.2f}M/min)  capped {capped}", flush=True)
        fout.close()
        print(f"[{month}] DONE {tok/1e6:.2f}M tokens, {docs} docs", flush=True)
    pool.shutdown()

    tot = 0
    for p in sorted(out_dir.glob("*.jsonl")):
        t = sum(json.loads(l).get("est_tokens", 0) for l in p.open() if l.strip())
        n = sum(1 for _ in p.open())
        tot += t
        print(f"  {p.name}: {n} docs, {t/1e6:.2f}M tokens")
    print(f"TOTAL ~{tot/1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
