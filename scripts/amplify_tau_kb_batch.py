"""Amplify the tau2 banking KB via OpenRouter's batch API, using a frontier generator.

Same prompts and grouping as amplify_tau_kb.py -- this only changes how the calls are dispatched.

Why a different generator: measured, not assumed. Auditing sampled documents against their exact
source pages (scripts/audit_tau_synth.py) found small open models assert an invented actionable
rule in 27% of documents -- fabricated MCC lists, a $1,000 sweep threshold, "dual review over
$10k". Prompt engineering plateaued at 13%. gpt-5.6-sol on the identical prompt: 5.7%, and of that
remainder most are judge artifacts or unit slips rather than invented policy. It also named zero
non-existent tools across ~300 documents, where the small models coined a whole plausible fake API.

That matters more here than it did for the news corpus. tau2 scores by final database state, so a
confabulated policy is not noise -- it is a wrong action that deterministically fails the task, and
it would leave us unable to tell a failed method from bad data.

The tool-name regex is retained purely as a tripwire. It is expected to catch nothing; if it starts
catching things, that is a signal about the generator, not a cleanup mechanism.

Batching is worth the extra machinery at this size: OpenRouter charges half rate for :batch models,
which on a 14M-token corpus is the difference between roughly $470 and $240.

    python scripts/amplify_tau_kb_batch.py --target-tokens 13.9e6 --out data/tau/kb-synth.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import time
from pathlib import Path

import requests

from amplify_tau_kb import DOC_TYPES, SEP, SYSTEM, TAIL, TOOL_RE, parse_docs

API = "https://openrouter.ai/api/beta/batches"


def post_batch(key: str, model: str, reqs: list, window: str) -> str:
    # The endpoint rejects a body whose `requests` key precedes `endpoint`/`model`, so the ordering
    # here is load-bearing and the body must be serialised by hand rather than passed as json=.
    body = collections.OrderedDict()
    body["endpoint"] = "/v1/chat/completions"
    body["model"] = model
    body["completion_window"] = window
    body["requests"] = reqs
    for attempt in range(5):
        r = requests.post(API, headers={"Authorization": f"Bearer {key}",
                                        "Content-Type": "application/json"},
                          data=json.dumps(body), timeout=600)
        if r.ok:
            return r.json()["id"]
        if r.status_code < 500 and r.status_code != 429:
            raise RuntimeError(f"batch submit failed {r.status_code}: {r.text[:300]}")
        time.sleep(min(120, 5 * 2 ** attempt))
    raise RuntimeError("batch submit failed after retries")


def poll(key: str, bid: str) -> dict:
    for attempt in range(6):
        r = requests.get(f"{API}/{bid}", headers={"Authorization": f"Bearer {key}"}, timeout=300)
        if r.ok:
            return r.json()
        time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"poll failed for {bid}")


def build_groups(kb: list, group_n: int, rnd: int, seed: int) -> list:
    """Regroup the pages afresh every round.

    A 68x amplification restates every page ~68 times; if the same three pages always travel
    together the generator sees the identical context each round and the corpus becomes 68 near
    copies. Re-pairing pages within their category each round varies the context itself, which
    varies the documents for free.
    """
    rng = random.Random(seed * 10_000 + rnd)
    by_cat: dict[str, list] = {}
    for d in kb:
        by_cat.setdefault("_".join(d["id"].split("_")[1:3]), []).append(d)
    groups = []
    for _cat, docs in by_cat.items():
        docs = docs[:]
        rng.shuffle(docs)
        for i in range(0, len(docs), group_n):
            groups.append(docs[i:i + group_n])
    rng.shuffle(groups)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="/tmp/tau2-bench/data/tau2/domains/banking_knowledge/documents")
    ap.add_argument("--out", default="data/tau/kb-synth.jsonl")
    ap.add_argument("--state", default=None, help="defaults to <out>.state.json")
    ap.add_argument("--target-tokens", type=float, default=13.9e6)
    ap.add_argument("--group-n", type=int, default=3)
    ap.add_argument("--per-call", type=int, default=7)
    ap.add_argument("--words", type=int, default=450)
    ap.add_argument("--model", default="openai/gpt-5.6-sol:batch")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--chunk", type=int, default=250, help="requests per batch job")
    ap.add_argument("--wave", type=int, default=4, help="batch jobs in flight per wave")
    ap.add_argument("--completion-window", default="24h")
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = os.environ[args.api_key_env]
    kb = [json.loads(p.read_text()) for p in sorted(Path(args.kb).iterdir())]
    all_tools: set[str] = set()
    for d in kb:
        all_tools |= set(TOOL_RE.findall(d["content"]))
    print(f"{len(kb)} KB documents, {len(all_tools)} distinct tool names")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state) if args.state else out_path.with_suffix(".state.json")
    state = {"round": 0, "tokens": 0, "docs": 0, "dropped": 0, "cost": 0.0, "done_ids": []}
    if state_path.exists():
        state.update(json.loads(state_path.read_text()))
        print(f"resume: round {state['round']}, {state['tokens']/1e6:.2f}M tokens, "
              f"{state['docs']} docs, ${state['cost']:.2f}")

    fout = out_path.open("a")
    t0 = time.time()

    while state["tokens"] < args.target_tokens:
        rnd = state["round"]
        groups = build_groups(kb, args.group_n, rnd, args.seed)
        reqs, meta = [], {}
        for gi, g in enumerate(groups):
            cid = f"{gi}#{rnd}"
            ctx = "\n\n".join(f"--- page: {d['title']} ---\n{d['content']}" for d in g)
            fmts = [DOC_TYPES[(rnd * args.per_call + k) % len(DOC_TYPES)]
                    for k in range(args.per_call)]
            src_tools = sorted(set(TOOL_RE.findall(ctx)))
            tail = TAIL.format(n=len(fmts), sep=SEP, words=args.words,
                               tools="\n".join(f"  - {t}" for t in src_tools)
                                     or "  (none -- name no tool)",
                               formats="\n".join(f"{i+1}. {f}" for i, f in enumerate(fmts)))
            reqs.append({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                         "body": {"max_completion_tokens": args.max_tokens,
                                  "temperature": 1.0, "top_p": 0.95,
                                  "messages": [
                                      {"role": "system", "content": SYSTEM.format(sep=SEP)},
                                      {"role": "user", "content": ctx + tail}]}})
            meta[cid] = [d["id"] for d in g]

        chunks = [reqs[i:i + args.chunk] for i in range(0, len(reqs), args.chunk)]
        chunks = chunks[: args.wave]
        ids = []
        for c in chunks:
            bid = post_batch(key, args.model, c, args.completion_window)
            ids.append(bid)
            print(f"  round {rnd}: submitted {bid} ({len(c)} requests)", flush=True)

        pending = set(ids)
        while pending:
            time.sleep(args.poll_seconds)
            for bid in list(pending):
                info = poll(key, bid)
                st = info.get("status")
                if st in ("completed", "failed", "cancelled", "expired"):
                    pending.discard(bid)
                    if st != "completed":
                        print(f"  {bid}: {st} -- {str(info.get('error'))[:200]}", flush=True)
                    n_new = 0
                    for res in info.get("results") or []:
                        resp = (res.get("response") or {}).get("body") or {}
                        ch = (resp.get("choices") or [{}])[0]
                        text = ((ch.get("message") or {}).get("content")) or ""
                        used = (resp.get("usage") or {}).get("completion_tokens") or len(text) // 4
                        cid = res.get("custom_id")
                        docs = parse_docs(text)
                        keep = []
                        for d in docs:
                            if set(TOOL_RE.findall(d)) - all_tools:
                                state["dropped"] += 1
                                continue
                            keep.append(d)
                        if not keep:
                            continue
                        share = max(1, used // len(keep))
                        for j, d in enumerate(keep):
                            fout.write(json.dumps(
                                {"call_id": cid, "doc_ix": j, "kind": "kb_synth",
                                 "gen": args.model, "est_tokens": share,
                                 "source_ids": meta.get(cid, []), "text": d},
                                ensure_ascii=False) + "\n")
                        state["tokens"] += used
                        state["docs"] += len(keep)
                        n_new += len(keep)
                    state["cost"] += float((info.get("usage") or {}).get("cost") or 0.0)
                    fout.flush()
                    el = time.time() - t0
                    print(f"  {bid}: {st}, +{n_new} docs | total {state['tokens']/1e6:.2f}M tok, "
                          f"{state['docs']} docs, ${state['cost']:.2f}, "
                          f"dropped {state['dropped']}, {el/60:.0f}m", flush=True)
                    state["done_ids"].append(bid)
                    state_path.write_text(json.dumps(state, indent=1))

        state["round"] += 1
        state_path.write_text(json.dumps(state, indent=1))

    fout.close()
    print(f"done: {state['tokens']/1e6:.2f}M tokens, {state['docs']} docs, "
          f"${state['cost']:.2f}, {state['dropped']} tripwire drops -> {args.out}")


if __name__ == "__main__":
    main()
