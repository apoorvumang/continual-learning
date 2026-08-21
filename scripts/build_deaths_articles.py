"""Turn the death index into real coverage: search for each death, fetch the articles.

The index built by build_deaths_corpus.py is one sentence per person -- "Arsenio Campos died on
April 1, 2025. 79, Mexican actor." That is the right *fact* but the wrong *document*: a stub teaches
almost nothing under continued pretraining, and it reads nothing like the prose the model will be
asked about.

So each index entry becomes a search: the name, windowed to the days after the death, through
keenable, then the articles fetched. A spot check found an obituary for an obscure British plant
breeder (3,829 characters) as readily as for a footballer, so coverage reaches well past the famous.

Relevance filtering is not optional. For one footballer the top-ranked result was an unrelated boxing
preview, so an article is kept only if it actually names the person -- surname match at minimum, and
a death word somewhere in the text.

Why this event class gets the extra effort: the knowledge-cutoff benchmark scores low-predictability
events, and deaths are its densest category. A model that has not learned them does not answer "I
don't know" -- it asserts the person is alive, which is the confidently-wrong failure the benchmark
treats as the strongest signal that an event postdates training.

    python scripts/build_deaths_articles.py --in data/news17/deaths \
        --out data/news17/deaths-articles --workers 8
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from keenable import fetch, search                              # noqa: E402

WS = re.compile(r"\s+")
DEATH_WORD = re.compile(r"\b(died|death|dies|passed away|obituary|dead|posthum)", re.I)
# Aggregators and list pages that mention many names and would pass a naive relevance check.
SKIP = re.compile(r"(wikipedia\.org|wikiwand|toolforge|dbpedia|everipedia|imdb\.com|"
                  r"findagrave|legacy\.com|dead-people|tributearchive|echovita|"
                  r"deaths-in-|list-of-deaths|obituaries\.com)", re.I)


def surname(name: str) -> str:
    parts = [p for p in re.split(r"\s+", name.strip()) if len(p) > 1]
    return parts[-1] if parts else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/news17/deaths")
    ap.add_argument("--out", default="data/news17/deaths-articles")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-death", type=int, default=2, help="articles kept per person")
    ap.add_argument("--consider", type=int, default=4, help="search results examined per person")
    ap.add_argument("--window-days", type=int, default=21)
    ap.add_argument("--min-chars", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    for src in sorted(Path(args.inp).glob("*.jsonl")):
        month = src.stem
        mp = out / f"{month}.jsonl"
        done = set()
        if mp.exists():
            for line in mp.open():
                if line.strip():
                    done.add(json.loads(line).get("person", ""))
        entries = [json.loads(l) for l in src.open() if l.strip()]
        todo = [e for e in entries if e["title"].replace("Death of ", "") not in done]
        if not todo:
            print(f"[{month}] complete, skip", flush=True)
            continue
        fout = mp.open("a")
        st = {"kept": 0, "people": 0, "nores": 0, "irrel": 0}

        def one(e):
            person = e["title"].replace("Death of ", "")
            sn = surname(person).lower()
            d = dt.date.fromisoformat(e["date"])
            try:
                res = search(f'"{person}" died',
                             published_after=str(d - dt.timedelta(days=1)),
                             published_before=str(d + dt.timedelta(days=args.window_days)))
            except Exception:                                    # noqa: BLE001
                res = []
            if not res:
                with lock:
                    st["nores"] += 1
                return
            got = 0
            for h in res[: args.consider]:
                url = h.get("url") or ""
                if not url or SKIP.search(url):
                    continue
                try:
                    doc = fetch(url)
                except Exception:                                # noqa: BLE001
                    continue
                txt = (doc.get("text") or doc.get("content") or "").strip()
                if len(txt) < args.min_chars:
                    continue
                low = txt.lower()
                # must actually be about this person's death
                if sn not in low or not DEATH_WORD.search(txt[:4000]):
                    with lock:
                        st["irrel"] += 1
                    continue
                with lock:
                    fout.write(json.dumps(
                        {"tier": "A", "date": e["date"], "url": url,
                         "domain": url.split("/")[2] if "://" in url else "",
                         "title": WS.sub(" ", (doc.get("title") or person)).strip(),
                         "text": txt[:20000], "est_tokens": min(len(txt), 20000) // 4,
                         "src": "deaths-article", "person": person}, ensure_ascii=False) + "\n")
                    st["kept"] += 1
                got += 1
                if got >= args.per_death:
                    break
            with lock:
                st["people"] += 1
                if st["people"] % 200 == 0:
                    fout.flush()
                    print(f"[{month}] {st['people']}/{len(todo)} people, {st['kept']} articles, "
                          f"{st['nores']} no-results, {st['irrel']} off-topic", flush=True)

        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(one, todo))
        fout.close()
        n = sum(1 for _ in mp.open())
        t = sum(json.loads(l).get("est_tokens", 0) for l in mp.open() if l.strip())
        print(f"[{month}] DONE {n} articles for {len(entries)} deaths, {t/1e6:.2f}M tokens",
              flush=True)


if __name__ == "__main__":
    main()
