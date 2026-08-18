"""Stage 2 of Active Reading via the batch API, because the live API cannot sustain it.

Measured on this key: 3 of 8 concurrent requests come back 429, and at concurrency 24 or 48 every
single one does, instantly. Live generation produced 34 documents in five minutes -- about seventy
hours for the corpus. The batch endpoint has no such limit, bills at half rate, and already produced
a 15.18M-token corpus for this project in roughly five hours.

Strategies come from `active_read_tau.py --stage strategies`; this only changes how the second stage
is dispatched. Batch quirks and hardening are carried over from amplify_tau_kb_batch.py, all of them
learned the hard way:

  * the endpoint rejects a body whose `requests` key precedes `endpoint`/`model`, so the ordering is
    load-bearing and the body must be serialised by hand
  * a poll failure is never a reason to abandon a batch that is already billed -- one DNS blip
    previously orphaned four in-flight batches
  * batch ids are recorded when ACCEPTED, not when collected, so they survive the process dying
  * jobs_for(round) is deterministic, which is what makes an orphaned batch adoptable: the results
    carry custom_ids and this recovers what each one was written from

    python scripts/active_read_batch.py --target-tokens 15e6 --out data/tau/ar-docs.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

import requests

from active_read_tau import APPLY_SYSTEM, APPLY_USER, TOOL_RE, build_groups, load_kb

API = "https://openrouter.ai/api/beta/batches"


def post_batch(key: str, model: str, reqs: list, window: str) -> str:
    body = collections.OrderedDict()
    body["endpoint"] = "/v1/chat/completions"
    body["model"] = model
    body["completion_window"] = window
    body["requests"] = reqs
    for attempt in range(8):
        try:
            r = requests.post(API, headers={"Authorization": f"Bearer {key}",
                                            "Content-Type": "application/json"},
                              data=json.dumps(body), timeout=600)
            if r.ok:
                return r.json()["id"]
            if r.status_code < 500 and r.status_code != 429:
                raise RuntimeError(f"batch submit failed {r.status_code}: {r.text[:300]}")
        except requests.RequestException as e:
            print(f"    submit network error, retrying: {str(e)[:110]}", flush=True)
        time.sleep(min(120, 5 * 2 ** attempt))
    raise RuntimeError("batch submit failed after retries")


def poll(key: str, bid: str) -> dict:
    attempt = 0
    while True:
        try:
            r = requests.get(f"{API}/{bid}", headers={"Authorization": f"Bearer {key}"},
                             timeout=300)
            if r.ok:
                return r.json()
            if r.status_code == 404:
                raise RuntimeError(f"batch {bid} does not exist")
        except requests.RequestException as e:
            if attempt % 10 == 0:
                print(f"    poll error on {bid}, still retrying: {str(e)[:110]}", flush=True)
        attempt += 1
        time.sleep(min(120, 5 * 2 ** min(attempt, 5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="/tmp/tau2-bench/data/tau2/domains/banking_knowledge/documents")
    ap.add_argument("--strategies", default="data/tau/ar-strategies.json")
    ap.add_argument("--out", default="data/tau/ar-docs.jsonl")
    ap.add_argument("--state", default=None)
    ap.add_argument("--target-tokens", type=float, default=15e6)
    ap.add_argument("--group-n", type=int, default=3)
    ap.add_argument("--words", type=int, default=450)
    ap.add_argument("--model", default="openai/gpt-5.6-sol:batch")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--wave", type=int, default=8, help="batch jobs in flight per round")
    ap.add_argument("--completion-window", default="24h")
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = os.environ[args.api_key_env]
    kb, all_tools, collide = load_kb(args.kb)
    store = json.loads(Path(args.strategies).read_text())
    skeys = sorted(store, key=lambda k: int(k))
    print(f"{len(kb)} KB pages, {len(all_tools)} tools; {len(skeys)} strategy groups, "
          f"{sum(len(store[k]['strategies']) for k in skeys)} strategies", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state) if args.state else out_path.with_suffix(".state.json")
    state = {"round": 0, "tokens": 0, "docs": 0, "dropped": 0, "cost": 0.0,
             "done_ids": [], "pending": []}
    if state_path.exists():
        state.update(json.loads(state_path.read_text()))
        state.setdefault("pending", [])
        print(f"resume: round {state['round']}, {state['tokens']/1e6:.2f}M tokens, "
              f"{state['docs']} docs, ${state['cost']:.2f}, "
              f"{len(state['pending'])} to adopt", flush=True)

    fout = out_path.open("a")
    t0 = time.time()

    def jobs_for(rnd: int):
        groups = build_groups(kb, args.group_n, rnd, args.seed)
        reqs, meta = [], {}
        for gi, g in enumerate(groups):
            ctx = "\n\n".join(f"--- page: {d['title']} ---\n{d['content']}" for d in g)
            src_tools = sorted(set(TOOL_RE.findall(ctx)))
            entry = store[skeys[gi % len(skeys)]]
            for si, strat in enumerate(entry["strategies"]):
                cid = f"{rnd}:{gi}:{si}"
                user = APPLY_USER.format(
                    ctx=ctx, strategy=strat, words=args.words,
                    tools="\n".join(f"  - {t}" for t in src_tools) or "  (none -- name no tool)")
                reqs.append({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                             "body": {"max_completion_tokens": args.max_tokens,
                                      "temperature": 1.0, "top_p": 0.95,
                                      "messages": [
                                          {"role": "system", "content": APPLY_SYSTEM},
                                          {"role": "user", "content": user}]}})
                meta[cid] = ([d["id"] for d in g], strat, src_tools)
        return reqs, meta

    def harvest(bid: str, rnd: int) -> None:
        _reqs, meta = jobs_for(rnd)
        while True:
            info = poll(key, bid)
            st = info.get("status")
            if st in ("completed", "failed", "cancelled", "expired"):
                break
            time.sleep(args.poll_seconds)
        if st != "completed":
            print(f"  {bid}: {st} -- {str(info.get('error'))[:180]}", flush=True)
        n_new = 0
        for res in info.get("results") or []:
            body = (res.get("response") or {}).get("body") or {}
            ch = (body.get("choices") or [{}])[0]
            text = ((ch.get("message") or {}).get("content") or "").strip()
            used = (body.get("usage") or {}).get("completion_tokens") or len(text) // 4
            cid = res.get("custom_id")
            src_ids, strat, src_tools = meta.get(cid, ([], "", []))
            if len(text) < 250:
                continue
            named = set(TOOL_RE.findall(text))
            if (named - all_tools) or ((named & collide) - set(src_tools)):
                state["dropped"] += 1
                continue
            fout.write(json.dumps({"job_id": cid, "kind": "ar_doc", "gen": args.model,
                                   "est_tokens": used, "strategy": strat,
                                   "source_ids": src_ids, "text": text},
                                  ensure_ascii=False) + "\n")
            state["tokens"] += used
            state["docs"] += 1
            n_new += 1
        state["cost"] += float((info.get("usage") or {}).get("cost") or 0.0)
        fout.flush()
        print(f"  {bid}: {st}, +{n_new} docs | {state['tokens']/1e6:.2f}M tok, "
              f"{state['docs']} docs, ${state['cost']:.2f}, dropped {state['dropped']}, "
              f"{(time.time()-t0)/60:.0f}m", flush=True)
        state["done_ids"].append(bid)
        state["pending"] = [p for p in state["pending"] if p["id"] != bid]
        state_path.write_text(json.dumps(state, indent=1))

    for p in list(state["pending"]):
        print(f"  adopting orphan {p['id']} (round {p['round']})", flush=True)
        harvest(p["id"], p["round"])

    while state["tokens"] < args.target_tokens:
        rnd = state["round"]
        reqs, _m = jobs_for(rnd)
        chunks = [reqs[i:i + args.chunk] for i in range(0, len(reqs), args.chunk)][: args.wave]
        ids = []
        for c in chunks:
            bid = post_batch(key, args.model, c, args.completion_window)
            ids.append(bid)
            state["pending"].append({"id": bid, "round": rnd})
            state_path.write_text(json.dumps(state, indent=1))
            print(f"  round {rnd}: submitted {bid} ({len(c)} requests)", flush=True)
        for bid in ids:
            harvest(bid, rnd)
            if state["tokens"] >= args.target_tokens:
                break
        state["round"] += 1
        state_path.write_text(json.dumps(state, indent=1))

    fout.close()
    print(f"done: {state['tokens']/1e6:.2f}M tokens, {state['docs']} docs, "
          f"${state['cost']:.2f}, {state['dropped']} dropped -> {args.out}")


if __name__ == "__main__":
    main()
