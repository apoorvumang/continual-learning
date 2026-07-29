"""Stage 1: pull real reporting for each SDF topic.

The SDF paper starts from a hand-written "universe context" describing the world in which
the target belief holds. Ours is post-cutoff *true* news, so the context can be grounded in
real articles instead of invented -- this stage collects them.

Source quality matters more than volume here: everything downstream (key facts, synthetic
documents) inherits errors from these articles, and the paper found consistency with the
universe context to be the critical ingredient. So wire services and major outlets are
preferred, syndication near-duplicates are dropped, and blog platforms are excluded.

    python scripts/retrieve_docs.py --out data/real
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from keenable import fetch, search

# Wire services and major outlets first; these get priority when filling the quota.
PREFERRED = {
    "apnews.com", "reuters.com", "bbc.com", "bbc.co.uk", "npr.org", "cnn.com",
    "nytimes.com", "washingtonpost.com", "theguardian.com", "aljazeera.com",
    "cbsnews.com", "nbcnews.com", "abcnews.go.com", "politico.com", "axios.com",
    "time.com", "usatoday.com", "newsweek.com", "thehill.com", "pbs.org",
    "japantimes.co.jp", "kyodonews.net", "nhk.or.jp", "asahi.com", "mainichi.jp",
    "timesofisrael.com", "jpost.com", "haaretz.com", "france24.com", "dw.com",
    "variety.com", "deadline.com", "hollywoodreporter.com", "billboard.com",
    "en.wikipedia.org", "sky.com", "news.sky.com", "independent.co.uk", "ft.com",
}
# User-generated / SEO-farm platforms: fluent but unreliable, and they inject opinion
# into what is supposed to be a factual universe context.
BLOCKED_SUBSTRINGS = ("substack.com", "medium.com", "blogspot.", "wordpress.com",
                      "reddit.com", "quora.com", "facebook.com", "x.com", "twitter.com",
                      "youtube.com", "tiktok.com", "pinterest.")

TOPICS = {
    "charlie-kirk": {
        "event_date": "2025-09-10",
        "event_ids": ["2025-09-death-charlie-kirk"],
        "queries": [
            "Charlie Kirk assassination Utah Valley University",
            "Charlie Kirk shot killed Turning Point USA",
            "Tyler Robinson charged Charlie Kirk shooting",
            "Charlie Kirk death reaction tributes",
            "Charlie Kirk shooting investigation details",
            "Charlie Kirk memorial service",
            "who was Charlie Kirk Turning Point USA founder",
        ],
    },
    "takaichi": {
        "event_date": "2025-10-21",
        "event_ids": ["2025-09-office-japan-ishiba-resign", "2025-10-office-japan-pm-takaichi"],
        "queries": [
            "Sanae Takaichi elected Prime Minister Japan",
            "Takaichi first female Japanese prime minister",
            "Shigeru Ishiba resignation announcement September 2025",
            "LDP leadership election October 2025 Takaichi",
            "Takaichi cabinet lineup policies",
            "Japan parliament vote prime minister Takaichi",
            "Takaichi coalition Komeito Ishin",
        ],
    },
    "khamenei": {
        "event_date": "2026-02-28",
        "event_ids": ["2026-02-death-khamenei", "2026-03-death-larijani",
                      "2026-03-office-iran", "2026-04-death-kharazi"],
        "queries": [
            "Ali Khamenei killed Israeli strike February 2026",
            "Iran Supreme Leader Khamenei death confirmed",
            "Khamenei successor named Assembly of Experts",
            "Ali Larijani killed Iran security council",
            "Kamal Kharazi died Iran adviser",
            "Iran leadership succession after Khamenei",
            "Israel strikes Iran February 2026 aftermath",
        ],
    },
}

WINDOW_DAYS = 60
PER_DOMAIN_CAP = 3
TARGET_DOCS = 40
MIN_CHARS = 600


def domain_of(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "").lower()


def usable(url: str) -> bool:
    d = domain_of(url)
    return not any(b in d for b in BLOCKED_SUBSTRINGS)


def norm_title(t: str) -> str:
    """Collapse titles so syndicated copies of one story dedupe to a single entry."""
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()[:70]


def collect(name: str, spec: dict) -> list[dict]:
    d0 = dt.date.fromisoformat(spec["event_date"])
    after = (d0 - dt.timedelta(days=3)).isoformat()
    before = (d0 + dt.timedelta(days=WINDOW_DAYS)).isoformat()

    hits: dict[str, dict] = {}
    for q in spec["queries"]:
        for r in search(q, published_after=after, published_before=before):
            if usable(r["url"]):
                hits.setdefault(r["url"], r)
    print(f"[{name}] {len(hits)} candidate urls from {len(spec['queries'])} queries")

    # preferred outlets first, then the rest; cap per domain and drop repeat headlines
    ranked = sorted(hits.values(),
                    key=lambda r: (domain_of(r["url"]) not in PREFERRED,
                                   r.get("published_at") or ""))
    picked, per_domain, titles = [], {}, set()
    for r in ranked:
        d = domain_of(r["url"])
        t = norm_title(r.get("title", ""))
        if per_domain.get(d, 0) >= PER_DOMAIN_CAP or (t and t in titles):
            continue
        picked.append(r)
        per_domain[d] = per_domain.get(d, 0) + 1
        titles.add(t)
        if len(picked) >= TARGET_DOCS:
            break

    def grab(r):
        try:
            page = fetch(r["url"])
        except Exception as e:
            return {"url": r["url"], "error": str(e)[:200]}
        return {
            "url": r["url"], "domain": domain_of(r["url"]),
            "title": page.get("title") or r.get("title"),
            "published_at": page.get("published_at") or r.get("published_at"),
            "author": page.get("author"),
            "content": page.get("content") or "",
            "preferred": domain_of(r["url"]) in PREFERRED,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        docs = list(pool.map(grab, picked))

    good = [d for d in docs if not d.get("error") and len(d.get("content", "")) >= MIN_CHARS]
    failed = len(docs) - len(good)
    chars = sum(len(d["content"]) for d in good)
    print(f"[{name}] fetched {len(good)} usable docs ({failed} unusable), "
          f"{chars/1000:.0f}k chars, {sum(d['preferred'] for d in good)} from preferred outlets")
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--topics", nargs="+", default=list(TOPICS))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.topics:
        docs = collect(name, TOPICS[name])
        payload = {"topic": name, **{k: v for k, v in TOPICS[name].items() if k != "queries"},
                   "n_docs": len(docs), "docs": docs}
        (out / f"{name}.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
        print(f"[{name}] wrote {out / f'{name}.json'}")


if __name__ == "__main__":
    main()
