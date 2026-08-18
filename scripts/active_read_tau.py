"""Active Reading over the tau2 banking KB: let the model decide how to study it.

Our first two corpora were paraphrase + synthetic QA under 14 hand-written document types. Meta's
Active Reading (arXiv 2508.09494) reports exactly those two methods plateauing in downstream recall
as tokens grow, while self-generated study strategies keep improving out to 4B words. We saw the
plateau: 15.2M tokens of documents left tau2 accuracy flat, and adding 10% Q/A lifted tool-name
recall 19.6% -> 78.3% while accuracy fell.

Their other finding is the one that indicts our design most directly: DIVERSITY drives learning, not
answer coverage. Synthetic QA had the highest coverage of target answers in their ablations and still
underperformed. We optimised coverage — all 698 pages, 92-98 documents each, every one of the 46
tools mentioned at least 83 times — and never measured diversity at all.

So this is two-stage, per their task-specific variant, which scored better than task-agnostic:

  stage 1  the model reads a group of KB pages, imagines what an agent will actually be asked to DO
           with them, and writes its own study strategies
  stage 2  each strategy is applied independently to produce one training document

The grounding machinery is kept from amplify_tau_kb.py, because it was necessary rather than
decorative: stating the closed tool set explicitly took invented tool names from 31% to 4% before the
generator was even changed, and the per-group check on collision-prone suffixes
(activate_debit_card_8291/_8292/_8293) catches the one substitution that is both plausible and
maximally wrong.

    python scripts/active_read_tau.py --stage strategies --out data/tau/ar-strategies.json
    python scripts/active_read_tau.py --stage docs --target-tokens 15e6 --out data/tau/ar-docs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

TOOL_RE = re.compile(r"\b([a-z][a-z_]*_\d{3,4})\b")
SEP = "<<<DOC>>>"

# Stage 1. Task-specific framing: imagine the downstream work first, then design study methods for
# it. The instruction to vary the *kind* of processing is deliberate -- if it returns fourteen
# rephrasings of "write a summary" we have reinvented our own fixed list with extra steps.
STRATEGY_SYSTEM = """You are about to study some pages from a bank's internal knowledge base so that
you can act as a support agent using them, from memory, without looking anything up.

First think about what you will actually be asked to DO with this material: which tool to call for a
described situation, which preconditions to check before acting, what order operations must happen
in, what to tell a customer who does not qualify, which of several similar-looking tools applies.

Then design {n} DIFFERENT strategies for studying this material so you will succeed at that.

Requirements for the strategies:
- Each must transform the material differently -- not {n} ways of saying "summarise it". Vary the
  kind of cognitive work: reorganise it, interrogate it, dramatise it, tabulate it, stress-test it,
  connect it to the rest of the domain, rehearse it as a procedure, compare confusable items.
- Each must be a concrete instruction you could hand to someone else and get a specific artifact.
- At least two must specifically drill the exact tool names and their preconditions, since those are
  arbitrary strings that cannot be guessed.
- At least one must work through what goes wrong if steps are done out of order.

Return JSON only:
{{"strategies": ["<instruction>", "<instruction>", ...]}}"""

# Stage 2. One strategy -> one document. Grounding rules carried over unchanged.
APPLY_SYSTEM = """You are studying pages from a bank's internal knowledge base by applying a specific
study strategy, and writing the result as a document that will be used as language-model training
data.

Hard rules:
- Use ONLY facts stated in the supplied pages. Invent no policy, threshold, fee, eligibility rule or
  time limit. If the material is thin on a point, leave it out.
- The ALLOWED TOOLS list is the COMPLETE set of tools that exist. Name no other tool, ever. Do not
  coin a plausible-sounding one. If a step has no tool, describe it in prose.
- Reproduce tool names EXACTLY, including the numeric suffix -- the suffix is part of the name, not a
  version number. Several tools differ only by suffix; getting one wrong is worse than omitting it.
- Preserve ordering constraints and the CONDITIONS on a rule, not just the rule.
- The document must stand alone: no reference to "the material", "the pages", "this strategy", or any
  source. Write it as the bank's own document.

Write the document and nothing else -- no preamble, no explanation of the strategy."""

APPLY_USER = """{ctx}

ALLOWED TOOLS -- the only tools that exist. Naming any other makes the document unusable:
{tools}

--------
STUDY STRATEGY to apply:
{strategy}

Produce the resulting document. Aim for roughly {words} words."""


def load_kb(path: str):
    kb = [json.loads(p.read_text()) for p in sorted(Path(path).iterdir())]
    all_tools: set = set()
    stems: dict = {}
    for d in kb:
        for t in TOOL_RE.findall(d["content"]):
            all_tools.add(t)
            stems.setdefault("_".join(t.split("_")[:-1]), set()).add(t)
    collide = {t for v in stems.values() if len(v) > 1 for t in v}
    return kb, all_tools, collide


def build_groups(kb: list, group_n: int, rnd: int, seed: int) -> list:
    """Regroup every round so the same pages do not always travel together.

    Fixed groupings would show the generator an identical context each round, which at high
    amplification produces near-duplicates -- the failure Active Reading is meant to avoid.
    """
    rng = random.Random(seed * 9973 + rnd)
    by_cat: dict = {}
    for d in kb:
        by_cat.setdefault("_".join(d["id"].split("_")[1:3]), []).append(d)
    groups = []
    for _c, docs in by_cat.items():
        docs = docs[:]
        rng.shuffle(docs)
        for i in range(0, len(docs), group_n):
            groups.append(docs[i:i + group_n])
    rng.shuffle(groups)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["strategies", "docs"], required=True)
    ap.add_argument("--kb", default="/tmp/tau2-bench/data/tau2/domains/banking_knowledge/documents")
    ap.add_argument("--strategies", default="data/tau/ar-strategies.json")
    ap.add_argument("--out", default="data/tau/ar-docs.jsonl")
    ap.add_argument("--target-tokens", type=float, default=15e6)
    ap.add_argument("--group-n", type=int, default=3)
    ap.add_argument("--n-strategies", type=int, default=10)
    ap.add_argument("--words", type=int, default=450)
    ap.add_argument("--model", default="openai/gpt-5.6-sol")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kb, all_tools, collide = load_kb(args.kb)
    print(f"{len(kb)} KB pages, {len(all_tools)} tools ({len(collide)} collision-prone)", flush=True)
    cl = openai.OpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env],
                       timeout=900, max_retries=4)
    lock = threading.Lock()

    # ---------------------------------------------------------------- stage 1
    if args.stage == "strategies":
        groups = build_groups(kb, args.group_n, 0, args.seed)
        out_path = Path(args.strategies)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        store = json.loads(out_path.read_text()) if out_path.exists() else {}
        todo = [(gi, g) for gi, g in enumerate(groups) if str(gi) not in store]
        print(f"{len(groups)} groups, {len(todo)} needing strategies", flush=True)
        st = {"done": 0, "fail": 0}

        def one(job):
            gi, g = job
            ctx = "\n\n".join(f"--- page: {d['title']} ---\n{d['content']}" for d in g)
            for a in range(4):
                try:
                    r = cl.chat.completions.create(
                        model=args.model, max_completion_tokens=2500, temperature=1.0,
                        response_format={"type": "json_object"},
                        messages=[{"role": "system",
                                   "content": STRATEGY_SYSTEM.format(n=args.n_strategies)},
                                  {"role": "user", "content": ctx}])
                    ss = json.loads(r.choices[0].message.content or "{}").get("strategies") or []
                    ss = [s.strip() for s in ss if isinstance(s, str) and len(s.strip()) > 25]
                    if len(ss) >= 3:
                        with lock:
                            store[str(gi)] = {"page_ids": [d["id"] for d in g], "strategies": ss}
                            st["done"] += 1
                            if st["done"] % 25 == 0:
                                out_path.write_text(json.dumps(store, indent=1))
                                print(f"  {st['done']} groups strategised", flush=True)
                        return
                except Exception:                                # noqa: BLE001
                    time.sleep(min(45, 3 * 2 ** a))
            with lock:
                st["fail"] += 1

        with ThreadPoolExecutor(args.concurrency) as ex:
            list(ex.map(one, todo))
        out_path.write_text(json.dumps(store, indent=1))
        n_s = sum(len(v["strategies"]) for v in store.values())
        print(f"done: {len(store)} groups, {n_s} strategies "
              f"({n_s/max(len(store),1):.1f}/group), {st['fail']} failed -> {out_path}")
        return

    # ---------------------------------------------------------------- stage 2
    store = json.loads(Path(args.strategies).read_text())
    kbid = {d["id"]: d for d in kb}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done, tokens_done = set(), 0
    if out_path.exists():
        for line in out_path.open():
            if line.strip():
                r = json.loads(line)
                done.add(r["job_id"])
                tokens_done += r.get("est_tokens", 0)
        print(f"resume: {len(done)} docs, ~{tokens_done/1e6:.2f}M tokens", flush=True)

    # Every (group, strategy) pair is one job. Cycle rounds until the token target is met, so a
    # strategy gets reapplied to freshly regrouped pages rather than to the same three every time.
    jobs = []
    rnd = 0
    est = 600
    need = max(1, int(1.15 * (args.target_tokens - tokens_done) / est))
    while len(jobs) < need and rnd < 60:
        groups = build_groups(kb, args.group_n, rnd, args.seed)
        for gi, g in enumerate(groups):
            entry = store.get(str(gi % len(store))) or store.get("0")
            for si, strat in enumerate(entry["strategies"]):
                jid = f"{rnd}:{gi}:{si}"
                if jid in done:
                    continue
                jobs.append((jid, g, strat))
                if len(jobs) >= need:
                    break
            if len(jobs) >= need:
                break
        rnd += 1
    print(f"{len(jobs)} (group x strategy) jobs queued for ~{args.target_tokens/1e6:.0f}M tokens",
          flush=True)

    fout = out_path.open("a")
    stt = {"tok": tokens_done, "docs": 0, "dropped": 0, "fail": 0, "t0": time.time()}

    def gen(job):
        jid, g, strat = job
        ctx = "\n\n".join(f"--- page: {d['title']} ---\n{d['content']}" for d in g)
        src_tools = sorted(set(TOOL_RE.findall(ctx)))
        user = APPLY_USER.format(ctx=ctx, strategy=strat, words=args.words,
                                 tools="\n".join(f"  - {t}" for t in src_tools)
                                       or "  (none -- name no tool)")
        text, used = "", 0
        for a in range(5):
            try:
                r = cl.chat.completions.create(
                    model=args.model, max_completion_tokens=args.max_tokens,
                    temperature=1.0, top_p=0.95,
                    messages=[{"role": "system", "content": APPLY_SYSTEM},
                              {"role": "user", "content": user}])
                text = (r.choices[0].message.content or "").strip()
                used = r.usage.completion_tokens if r.usage else len(text) // 4
                break
            except Exception:                                    # noqa: BLE001
                time.sleep(min(60, 3 * 2 ** a))
        if len(text) < 250:
            with lock:
                stt["fail"] += 1
            return
        named = set(TOOL_RE.findall(text))
        # Whole-KB for ordinary tools; own-group provenance for the collision-prone stems.
        if (named - all_tools) or ((named & collide) - set(src_tools)):
            with lock:
                stt["dropped"] += 1
            return
        with lock:
            fout.write(json.dumps({"job_id": jid, "kind": "ar_doc", "gen": args.model,
                                   "est_tokens": used, "strategy": strat,
                                   "source_ids": [d["id"] for d in g], "text": text},
                                  ensure_ascii=False) + "\n")
            stt["tok"] += used
            stt["docs"] += 1
            if stt["docs"] % 200 == 0:
                fout.flush()
                el = time.time() - stt["t0"]
                print(f"  {stt['tok']/1e6:.2f}M tok  {stt['docs']} docs  "
                      f"{(stt['tok']-tokens_done)/max(el,1):.0f} tok/s  "
                      f"dropped {stt['dropped']}  fails {stt['fail']}", flush=True)

    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(gen, jobs))
    fout.close()
    print(f"done: ~{stt['tok']/1e6:.2f}M tokens, {stt['docs']} docs, "
          f"{stt['dropped']} dropped, {stt['fail']} failed -> {args.out}")


if __name__ == "__main__":
    main()
