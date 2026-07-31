"""Does injected knowledge make a model a cheaper search agent?

Anirudh's claim: on a k-hop question, a model that already knows the first k-1 hops should
search once instead of k times. This measures it -- same questions, same tool, same loop.

Result so far: no. See eval/searchqa/README.md. With a live web search tool no run ever used
more than ONE search, because real search APIs return summarised snippets that collapse a
2-hop entity-bridge question into a single query. Stale knowledge does not cause extra
searching, it causes *no* searching: 0 searches -> 0% correct, 1 search -> ~91%.

Two tools:
  --tool web     live keenable search, cached per query so repeats are free and every model
                 sees identical results. Use with eval/searchqa/ceo_hops.json, whose hop-1
                 facts (CEO changes) are all post-cutoff.
  --tool corpus  BM25 over the 120 articles in data/real/. Used with chains.json, where the
                 early hops are in the SDF training corpus and the final hop is verified
                 absent from it -- so a trained model can skip ahead but must still retrieve
                 to finish. Without that asymmetry a trained model just recites the answer,
                 search count goes to zero, and the measurement says nothing about search.

Worth knowing what published benchmarks do here: WebDetective (arXiv 2510.05137) screens
candidates for "parametric inaccessibility" and discards any the model can answer from
memory, and Exa's WebCode drops candidates the base model already knew. Existing evals treat
the parametric shortcut as contamination. This asks whether the shortcut is useful, which is
only a fair question if accuracy is reported next to the saving.

    python scripts/search_agent.py --tool web --chains eval/searchqa/ceo_hops.json \
        --model moe-kirk --repeats 4 --out eval/searchqa/ceo-35B-stale.json
    python scripts/search_agent.py --compare eval/searchqa/ceo-*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

CHUNK, OVERLAP, TOPK, MAX_SEARCH = 500, 100, 3, 6

SYSTEM = """You are answering a question using a search tool over a corpus of news articles.

Reply with EXACTLY ONE of these two lines each turn:
SEARCH: <query>
ANSWER: <your final answer>

Rules:
- Use SEARCH when you need information you do not already know.
- If you already know part of the answer, do not search for that part.
- Keep queries short, like search-engine keywords.
- You have at most {max_search} searches. Answer as soon as you can.
- ANSWER must be short: just the name, title, or phrase asked for."""

# The line telling the model not to search for what it thinks it knows is the obvious
# confound for any "it did not search" finding, so the same run is available without it.
SYSTEM_NEUTRAL = """You are answering a question using a search tool over the live web.

Reply with EXACTLY ONE of these two lines each turn:
SEARCH: <query>
ANSWER: <your final answer>

Rules:
- The question may concern recent events. Verify facts with SEARCH before answering.
- Keep queries short, like search-engine keywords.
- You have at most {max_search} searches.
- ANSWER must be short: just the name, title, or phrase asked for."""


# ---------------------------------------------------------------- retrieval
def tokenize(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


class BM25:
    """Textbook BM25. Written out rather than imported so the tool has no hidden behaviour."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.toks = [tokenize(c["text"]) for c in chunks]
        self.len = [len(t) for t in self.toks]
        self.avg = sum(self.len) / max(1, len(self.len))
        self.tf = [Counter(t) for t in self.toks]
        df = Counter(tok for t in self.toks for tok in set(t))
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query: str, topk: int = TOPK) -> list[dict]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                if w not in tf:
                    continue
                f, dl = tf[w], self.len[i]
                s += self.idf.get(w, 0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:topk]]


class WebSearchTool:
    """Live web search via keenable, which is what a real search agent actually calls.

    Results are cached to disk by query string. Two reasons, both load-bearing: it keeps us
    far under the rate limit across repeats, and it means two models issuing the same query
    see byte-identical results, so a search-count comparison is not confounded by the index
    shifting under us mid-run.
    """

    def __init__(self, cache_path="eval/searchqa/websearch-cache.json", topk=5):
        from keenable import search as _search
        self._search, self.topk = _search, topk
        self.path = Path(cache_path)
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.hits = self.misses = 0

    def search(self, query: str, topk: int | None = None) -> list[dict]:
        k = topk or self.topk
        if query not in self.cache:
            self.misses += 1
            try:
                res = self._search(query)
            except Exception as e:
                res = [{"title": f"search error: {str(e)[:80]}", "snippet": ""}]
            self.cache[query] = [{"title": r.get("title") or "",
                                  "text": (r.get("snippet") or r.get("description") or "")}
                                 for r in res[:10]]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache, indent=0, ensure_ascii=False))
        else:
            self.hits += 1
        return [{"domain": h["title"][:60], "text": h["text"]}
                for h in self.cache[query][:k]]


def load_corpus(real_dir="data/real") -> list[dict]:
    chunks = []
    for p in sorted(glob.glob(f"{real_dir}/*.json")):
        d = json.load(open(p))
        for doc in d["docs"]:
            text = " ".join(doc["content"].split())
            for i in range(0, max(1, len(text) - OVERLAP), CHUNK - OVERLAP):
                piece = text[i:i + CHUNK]
                if len(piece) > 120:
                    chunks.append({"topic": d["topic"], "domain": doc["domain"],
                                   "title": doc.get("title") or "", "text": piece})
    return chunks


# ---------------------------------------------------------------- agent loop
def run_chain(client, model, bm25, chain, max_search=MAX_SEARCH, thinking=False,
              system=SYSTEM) -> dict:
    msgs = [{"role": "system", "content": system.format(max_search=max_search)},
            {"role": "user", "content": chain["question"]}]
    queries, transcript = [], []
    answer, stop = None, "ok"

    for _ in range(max_search + 1):
        r = client.chat.completions.create(
            model=model, messages=msgs, max_completion_tokens=2048 if thinking else 200,
            temperature=0.7, top_p=0.8, presence_penalty=1.5,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": thinking}})
        out = (r.choices[0].message.content or "").strip()
        transcript.append({"role": "assistant", "text": out})

        m = re.search(r"ANSWER:\s*(.+)", out)
        if m:
            answer = m.group(1).strip()
            break
        m = re.search(r"SEARCH:\s*(.+)", out)
        if not m:                                  # protocol violation: treat text as answer
            answer, stop = out, "no-protocol"
            break
        if len(queries) >= max_search:
            stop = "budget"
            msgs.append({"role": "assistant", "content": out})
            msgs.append({"role": "user", "content":
                         "Search budget exhausted. Reply now with ANSWER: <answer>."})
            continue

        q = m.group(1).strip()
        queries.append(q)
        hits = bm25.search(q)
        obs = "\n\n".join(f"[{h['domain']}] {h['text']}" for h in hits) or "No results."
        msgs.append({"role": "assistant", "content": out})
        msgs.append({"role": "user", "content": f"Results:\n{obs}"})
        transcript.append({"role": "tool", "query": q,
                           "returned": [h["text"][:200] for h in hits]})

    hay = (answer or "").lower()
    # Both hops must be present. Checking only `gold` silently passes an answer that names the
    # outgoing CEO on the chains where the hop-2 answer IS the predecessor (nike, alstom): the
    # model resolves neither hop and still matches the gold string.
    got_gold = any(g.lower() in hay for g in chain["gold"])
    got_bridge = (any(b.lower() in hay for b in chain["bridge"])
                  if chain.get("bridge") else True)
    correct = got_gold and got_bridge
    # The mechanism test: does the FIRST query already name the bridge entity the question
    # withheld? If so the model resolved hop 1 from weights rather than by searching for it.
    first = (queries[0].lower() if queries else "")
    allq = " ".join(queries).lower()
    skipped = bool(queries) and any(b.lower() in first for b in chain["bridge"])
    # ...and does any query name the *predecessor*? That is searching from a stale premise,
    # the specific waste continued pretraining would remove.
    stale = [s for s in chain.get("stale", []) if s.lower() in allq]
    return {"id": chain["id"], "hops": chain["hops"], "n_searches": len(queries),
            "queries": queries, "answer": answer, "correct": correct,
            "got_bridge": got_bridge, "got_gold": got_gold,
            "bridge_in_first_query": skipped, "stale_in_queries": stale,
            "answer_has_stale": any(s.lower() in hay for s in chain.get("stale", [])),
            "no_search": len(queries) == 0, "stop": stop, "transcript": transcript}


# ---------------------------------------------------------------- corpus diagnostics
def leak_rate(bm25, chains) -> dict:
    """How often does a naive hop-1 query already retrieve the final answer? If this is high
    the corpus, not the model, is collapsing the hops -- report it rather than discover it
    later."""
    out = {}
    for c in chains:
        hits = bm25.search(c["question"])
        blob = " ".join(h["text"].lower() for h in hits)
        out[c["id"]] = any(t.lower() in blob for t in c["leak_terms"])
    return out


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["correct"]]
    return {
        "accuracy": [len(ok), len(rows)],
        "mean_searches_all": round(sum(r["n_searches"] for r in rows) / max(1, len(rows)), 2),
        "mean_searches_correct": (round(sum(r["n_searches"] for r in ok) / len(ok), 2)
                                  if ok else None),
        "bridge_in_first_query": [sum(r["bridge_in_first_query"] for r in rows), len(rows)],
        "stale_in_queries": [sum(bool(r.get("stale_in_queries")) for r in rows), len(rows)],
        "answer_has_stale": [sum(r.get("answer_has_stale", False) for r in rows), len(rows)],
        "answered_without_search": [sum(r["no_search"] for r in rows), len(rows)],
        "protocol_violations": sum(r["stop"] == "no-protocol" for r in rows),
        "budget_exhausted": sum(r["stop"] == "budget" for r in rows),
    }


def table(reports: list[dict]) -> str:
    labels = [r["label"] for r in reports]
    w = max(len(l) for l in labels + ["metric"]) + 3
    rows = [("accuracy", "accuracy"), ("mean_searches_all", "mean searches (all)"),
            ("mean_searches_correct", "mean searches (correct only)"),
            ("bridge_in_first_query", "bridge entity in 1st query"),
            ("stale_in_queries", "searched the stale entity"),
            ("answer_has_stale", "answered with stale entity"),
            ("answered_without_search", "answered with 0 searches"),
            ("protocol_violations", "protocol violations")]
    out = ["metric".ljust(32) + "".join(l.ljust(w) for l in labels),
           "-" * (32 + w * len(labels))]
    for key, name in rows:
        cells = []
        for r in reports:
            v = r["summary"][key]
            cells.append((f"{v[0]}/{v[1]}" if isinstance(v, list)
                          else ("-" if v is None else str(v))).ljust(w))
        out.append(name.ljust(32) + "".join(cells))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--chains", default="eval/searchqa/chains.json")
    ap.add_argument("--tool", choices=["corpus", "web"], default="corpus",
                    help="corpus = BM25 over data/real/; web = live keenable search (cached)")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--max-search", type=int, default=MAX_SEARCH)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--neutral-prompt", action="store_true",
                    help="drop the 'do not search what you know' line")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs="+", default=None)
    args = ap.parse_args()

    if args.compare:
        print(table([json.loads(Path(p).read_text()) for p in args.compare]))
        return

    chains = json.load(open(args.chains))["chains"]
    if args.tool == "web":
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        bm25 = WebSearchTool()
        leaks = {}
        print(f"tool: live web search (keenable), {len(bm25.cache)} queries cached")
    else:
        corpus = load_corpus()
        bm25 = BM25(corpus)
        print(f"corpus: {len(corpus)} chunks of {CHUNK} chars from data/real/")
        leaks = leak_rate(bm25, chains)
        print(f"hop-1-query leak: {sum(leaks.values())}/{len(leaks)} chains "
              f"({', '.join(k for k, v in leaks.items() if v) or 'none'})")

    client = openai.OpenAI(base_url=args.base_url, api_key="local")
    jobs = [c for c in chains for _ in range(args.repeats)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda c: run_chain(
            client, args.model, bm25, c, args.max_search, args.thinking,
            SYSTEM_NEUTRAL if args.neutral_prompt else SYSTEM), jobs))

    rep = {"label": args.label or args.model, "model": args.model,
           "repeats": args.repeats, "leak": leaks,
           "summary": summarize(rows), "rows": rows}
    print()
    print(table([rep]))
    print("\nper chain:")
    for c in chains:
        rs = [r for r in rows if r["id"] == c["id"]]
        acc = sum(r["correct"] for r in rs)
        sk = sum(r["bridge_in_first_query"] for r in rs)
        ns = sum(r["n_searches"] for r in rs) / len(rs)
        print(f"  {c['id']:28s} {c['hops']}hop  correct {acc}/{len(rs)}  "
              f"searches {ns:.1f}  bridge-in-1st {sk}/{len(rs)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
