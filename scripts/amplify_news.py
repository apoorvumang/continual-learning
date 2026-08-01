"""Amplify the real 2026 news corpus into ~100M synthetic tokens, grounded per document.

Why amplify at all. Arm A trained on 4.17M tokens of real news and barely fit it (loss
1.878 -> 1.695) because real reporting states a fact once or twice, where the SDF runs
repeated each fact thousands of times and got strong injection. So the two runs so far sit at
opposite corners:

    SDF, one topic, heavy repetition    strong injection, 100% fabrication
    real news, broad, no repetition     weak injection,   0% fabrication

This fills the untested corner: broad corpus AND heavy repetition. If the fabrication stays
at zero while injection rises, the recipe is "diverse corpus, amplified" and neither property
has to be traded for the other.

Grounding is the whole safety property. Every synthetic document is written from real articles
held in context, and the prompt forbids facts not present in them. Ungrounded generation at 24x
would launder the generator's own stale knowledge into 100M tokens of training data.

The unit of amplification is a GROUP of ~5 same-day articles, not a single article. Measured:
one article is a 775-token wire brief, and asking for 24 documents from three sentences produced
near-identical paraphrases -- the source, not the prompt, was the limit. A group carries ~7k
tokens covering several distinct events, which supports genuinely different documents.

Jobs are ordered so the group context is a stable prompt prefix, which vllm's prefix cache
serves nearly free (run the server with --enable-prefix-caching); the varying instruction goes
at the end. Without that the ~7k input would be re-prefilled for every call.

Output carries `date` and `source_url`, so the Jan-May / Jun-Jul split still applies and any
synthetic document can be traced to the article it came from.

    python scripts/amplify_news.py --target-tokens 100e6 --date-max 2026-05-31
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

SEP = "<<<DOC>>>"

# Rotating so no single register dominates. The SDF run's corpus was entirely "prominent
# person dies" reporting and the model learned the genre along with the facts; a format list
# this wide is the cheapest insurance against repeating that at 24x scale.
DOC_TYPES = [
    "a straight news report from a wire service",
    "a short wire brief, five sentences at most",
    "an analysis piece explaining why this matters",
    "an explainer answering the basic questions a reader would have",
    "an encyclopedia-style reference entry",
    "a dated timeline of how events unfolded",
    "a question-and-answer FAQ",
    "an excerpt from a live blog, with timestamps",
    "a background profile of the main people or organisations involved",
    "a local newspaper's coverage, written for readers near the events",
    "a broadcast script as a news anchor would read it",
    "an email newsletter summarising the day for subscribers",
    "a fact-check assessing claims made about the events",
    "a retrospective written some weeks later",
    "a briefing memo for someone who needs to be caught up quickly",
    "an editorial arguing a position about the events",
    "a sober academic note situating the events in context",
    "a market or business note on the financial implications",
    "a transcript excerpt of two correspondents discussing the events",
    "a reader's letter to an editor responding to the coverage",
    "a summary written for a general international audience",
    "a detailed follow-up report adding context to the original story",
]

SYSTEM = """You write realistic documents about news events, for use as language-model
training data.

You will be given several real news reports from one day. Write the requested documents about
the events they describe.

Hard rules:
- Use ONLY facts stated in the supplied material. Do not add events, numbers, names, quotes or
  dates that are not there. If the material on some event is thin, cover a different event
  rather than inventing detail.
- State the date of the events explicitly where it is natural to do so.
- Each document must stand alone, and must not refer to "the article" or "the source".
- Vary length, sentence structure and vocabulary between documents. Do not paraphrase the same
  sentences repeatedly.
- No headings like "Document 1". Just write each document.

Separate consecutive documents with a line containing exactly {sep}"""

TAIL = """

--------
Now write {n} DIFFERENT documents about the events above, in these formats, in order:
{formats}

Spread them across DIFFERENT events from the material above rather than all covering the same
one. Aim for roughly {words} words per document.

Separate consecutive documents with a line containing exactly {sep}"""


def parse_docs(text: str) -> list[str]:
    out = []
    for piece in text.split(SEP):
        p = piece.strip()
        p = re.sub(r"^(?:Document\s*\d+\s*[:.\-]?\s*)", "", p, flags=re.I)
        if len(p) > 250:                     # drop stubs and stray preamble
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/news2026/docs.jsonl")
    ap.add_argument("--out", default="data/news2026/synth.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--model", default="stock")
    ap.add_argument("--target-tokens", type=float, default=100e6)
    # Prefill dominates: the ~7k group context is re-prefilled every call because
    # prefix caching is unavailable on this hybrid model (recurrent GDN state cannot
    # be cached like KV -- vllm silently ignores --enable-prefix-caching). So ask for
    # many, longer documents per call to amortise that input over more output.
    ap.add_argument("--per-call", type=int, default=12)
    ap.add_argument("--words", type=int, default=550)
    ap.add_argument("--group-n", type=int, default=5,
                    help="real articles per group; the shared prompt prefix")
    ap.add_argument("--max-tokens", type=int, default=14000)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--date-min", default=None)
    ap.add_argument("--date-max", default="2026-05-31")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by_day = {}
    for line in Path(args.docs).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        d = r.get("date", "")
        if (args.date_min and d < args.date_min) or (args.date_max and d > args.date_max):
            continue
        if len(r["text"]) > 400:
            by_day.setdefault(d, []).append(r)

    # Groups of GROUP_N same-day articles. The day's Wikipedia summary leads every group: it
    # names all of that day's events in a few hundred tokens, which is what lets a document
    # place its subject in context instead of restating one wire brief.
    groups = []
    for d in sorted(by_day):
        docs = by_day[d]
        summary = next((x["text"] for x in docs if x.get("kind") == "summary"), "")
        arts = [x for x in docs if x.get("kind") != "summary"]
        for i in range(0, max(1, len(arts)), args.group_n):
            chunk = arts[i:i + args.group_n]
            if not chunk:
                continue
            # Kept tight on purpose. Prefill is ~60% of the compute at 24x amplification,
            # so every character of context is paid again on every call.
            ctx = ("Events of " + d + "\n\n" + summary[:2500] + "\n\n"
                   + "\n\n".join("---\n" + a["text"][:2000] for a in chunk))
            groups.append({"date": d, "gid": f"{d}#{i // args.group_n}",
                           "ctx": ctx, "urls": [a.get("url", "") for a in chunk]})
    print(f"{sum(len(v) for v in by_day.values())} source documents over {len(by_day)} days "
          f"-> {len(groups)} groups of <= {args.group_n} articles "
          f"(~{sum(len(g['ctx']) for g in groups)//max(1,len(groups))//4} tokens of context each)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_calls: set[str] = set()
    tokens_done = 0
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done_calls.add(r["call_id"])
                tokens_done += r["est_tokens"]
        print(f"resume: {len(done_calls)} calls already done, ~{tokens_done/1e6:.1f}M tokens")

    # Rounds over the group list until the token target is met. Jobs stay grouped by group id
    # and in order, NOT shuffled, so consecutive calls share a prompt prefix and the server's
    # prefix cache serves the ~7k context instead of re-prefilling it.
    est_per_call = args.per_call * int(args.words * 1.4)
    n_calls = max(1, int((args.target_tokens - tokens_done) / est_per_call) + 1)
    rounds = n_calls // max(1, len(groups)) + 1
    jobs = []
    for g in groups:
        for rd in range(rounds):
            fmts = [DOC_TYPES[(rd * args.per_call + k) % len(DOC_TYPES)]
                    for k in range(args.per_call)]
            cid = f"{g['gid']}#{rd}"
            if cid not in done_calls:
                jobs.append((cid, g, fmts))
    print(f"{len(jobs)} calls queued ({rounds} rounds x {len(groups)} groups), "
          f"{args.per_call} documents each, targeting "
          f"{args.target_tokens/1e6:.0f}M tokens")

    client = openai.OpenAI(base_url=args.base_url, api_key="local", timeout=600)
    lock = threading.Lock()
    fout = out_path.open("a")
    state = {"tok": tokens_done, "calls": 0, "docs": 0, "fail": 0, "t0": time.time()}
    stop = threading.Event()

    def one(job):
        cid, g, fmts = job
        if stop.is_set():
            return
        tail = TAIL.format(n=len(fmts), sep=SEP, words=args.words,
                           formats="\n".join(f"{i+1}. {f}" for i, f in enumerate(fmts)))
        try:
            r = client.chat.completions.create(
                model=args.model, max_completion_tokens=args.max_tokens,
                messages=[{"role": "system", "content": SYSTEM.format(sep=SEP)},
                          {"role": "user", "content": g["ctx"] + tail}],
                temperature=1.0, top_p=0.95,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
            text = r.choices[0].message.content or ""
            used = r.usage.completion_tokens if r.usage else len(text) // 4
        except Exception as e:
            with lock:
                state["fail"] += 1
                if state["fail"] % 25 == 1:
                    print(f"  gen error: {str(e)[:110]}", flush=True)
            return
        docs = parse_docs(text)
        if not docs:
            with lock:
                state["fail"] += 1
            return
        share = max(1, used // len(docs))
        with lock:
            for j, d in enumerate(docs):
                fout.write(json.dumps({
                    "call_id": cid, "doc_ix": j, "kind": "synth",
                    "date": g["date"], "group": g["gid"], "source_urls": g["urls"],
                    "est_tokens": share, "text": d}, ensure_ascii=False) + "\n")
            state["tok"] += used
            state["calls"] += 1
            state["docs"] += len(docs)
            if state["calls"] % 100 == 0:
                fout.flush()
                el = time.time() - state["t0"]
                rate = (state["tok"] - tokens_done) / max(el, 1)
                left = max(0.0, args.target_tokens - state["tok"]) / max(rate, 1)
                print(f"{state['tok']/1e6:6.1f}M tokens  {state['docs']:7d} docs  "
                      f"{rate:7.0f} tok/s  eta {left/3600:4.1f}h  fails {state['fail']}",
                      flush=True)
            if state["tok"] >= args.target_tokens:
                stop.set()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(one, jobs))
    fout.close()
    print(f"done: ~{state['tok']/1e6:.1f}M tokens, {state['docs']} documents, "
          f"{state['fail']} failed calls -> {out_path}")


if __name__ == "__main__":
    main()
