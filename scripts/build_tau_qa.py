"""Make the injected knowledge reachable, not just present.

The measured problem: after training, verbatim recall of tau2's 46 tool names is 41% when the model
is prompted in the DOCTAG format it was trained in, and 19.6% when simply asked a question. Half of
what got stored cannot be reached in the shape an agent needs it. That is not a data-quantity
problem -- coverage was uniform, and even the rarest tool appeared 83 times across 23,212 documents.
It is a directionality problem, and it is the reversal curse wearing a different hat: the model can
continue a document containing close_bank_account_7392 but cannot answer "which tool closes an
account".

So generate the reverse direction explicitly, from the REAL KB pages rather than the synthetic
corpus -- the source is only 204K tokens and is the one text guaranteed free of generator error, so
there is no reason to paraphrase a paraphrase.

Three kinds, and the first is the one that matters most:

  function -> name   "Which tool closes a bank account?" -> close_bank_account_7392.
                     This is exactly the query an agent issues internally, and exactly what the
                     recall probe measures at 19.6%.
  name -> function   "What does close_bank_account_7392 do?" Cheap, and it anchors the string as an
                     object of knowledge rather than a token that only appears mid-sentence.
  policy             conditions, thresholds and ordering. Tool names are not the only arbitrary
                     knowledge here; "file the dispute before requesting the limit increase" is
                     equally unguessable and equally graded.

Deliberately kept to a modest share of the corpus. The earlier format sweep found a QA-heavy mix
(25% of budget) scored BELOW bare documents -- but that measured storage via answer logprob, and
this is being added for accessibility, which that sweep never measured. Mixing a little is the
experiment; mixing a lot is repeating a known mistake.

    python scripts/build_tau_qa.py --target-tokens 2e6
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
SEP = "<<<QA>>>"

SYSTEM = """You write question-and-answer pairs for training a bank's customer-support agent.

You are given real pages from the bank's knowledge base. Write Q/A pairs answerable ENTIRELY from
those pages.

Hard rules:
- Use ONLY facts stated in the supplied pages. Invent no policy, threshold, fee, condition or tool.
- Reproduce tool names EXACTLY, including the numeric suffix. The suffix is part of the name, never
  a version number, and never to be renumbered or omitted.
- Ask the question the way someone would who does NOT already know the answer. "Which tool applies
  a statement credit?" is useful. "What does apply_statement_credit_8472 do?" is only useful
  sometimes, and the mix below says when.
- Each answer must stand alone: no "as stated above", no reference to any page or document.
- The answer states the fact directly and completely, in one to three sentences.

Format each pair as exactly:
Q: <question>
A: <answer>

Separate consecutive pairs with a line containing exactly {sep}"""

TAIL = """

--------
Write {n} Q/A pairs about the pages above, in this mix:
{mix}

Separate consecutive pairs with {sep}"""

# Two mixes, chosen per group by what the pages actually contain. A fixed tool-heavy mix against
# tool-free pages makes the generator refuse outright -- correctly, since answering would mean
# inventing a tool name -- and that silently failed 3 of every 4 calls in the first version.
TOOL_KINDS = (["Ask which TOOL performs a described action, so the answer is the exact tool name "
               "(include the numeric suffix)."] * 5 +
              ["Ask what a specific named tool does, so the answer describes its purpose and "
               "preconditions."] * 2 +
              ["Ask about an eligibility condition or precondition on using that tool."] * 2 +
              ["Ask about ordering: what must happen before the tool is called, and why."])

POLICY_KINDS = (["Ask about an eligibility condition, threshold, fee or limit."] * 3 +
                ["Ask about ordering: what must happen before what, and why."] * 2 +
                ["Ask what to do when a customer does NOT qualify, or the action is blocked."] * 2 +
                ["Ask a practical how-do-I question a support agent would have."] * 2)


def parse_pairs(text: str) -> list:
    out = []
    for chunk in text.split(SEP):
        m = re.search(r"Q:\s*(.+?)\n+A:\s*(.+)", chunk.strip(), re.S)
        if not m:
            continue
        q, a = m.group(1).strip(), m.group(2).strip()
        if len(q) > 10 and len(a) > 15:
            out.append((q, a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="/tmp/tau2-bench/data/tau2/domains/banking_knowledge/documents")
    ap.add_argument("--out", default="data/tau/kb-qa.jsonl")
    ap.add_argument("--target-tokens", type=float, default=2e6)
    ap.add_argument("--group-n", type=int, default=2)
    ap.add_argument("--per-call", type=int, default=12)
    ap.add_argument("--model", default="openai/gpt-5.6-sol")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--tool-share", type=float, default=0.6,
                    help="fraction of calls drawn from the 44 tool-bearing pages")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kb = [json.loads(p.read_text()) for p in sorted(Path(args.kb).iterdir())]
    all_tools: set[str] = set()
    stems: dict[str, set] = {}
    for d in kb:
        for t in TOOL_RE.findall(d["content"]):
            all_tools.add(t)
            stems.setdefault("_".join(t.split("_")[:-1]), set()).add(t)
    collide = {t for v in stems.values() if len(v) > 1 for t in v}
    print(f"{len(kb)} KB pages, {len(all_tools)} tools ({len(collide)} collision-prone)")

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    tokens_done = 0
    if out_path.exists():
        for line in out_path.open():
            if line.strip():
                r = json.loads(line)
                done.add(r["call_id"])
                tokens_done += r.get("est_tokens", 0)
        print(f"resume: {len(done)} calls, ~{tokens_done/1e6:.2f}M tokens")

    # Only 44 of 698 pages name a tool -- 6% -- yet those 46 strings are what the benchmark
    # actually grades, and uniform sampling buries them. Oversample them hard.
    tool_pages = [d for d in kb if TOOL_RE.findall(d["content"])]
    other_pages = [d for d in kb if not TOOL_RE.findall(d["content"])]
    print(f"{len(tool_pages)} pages name a tool, {len(other_pages)} do not; "
          f"{args.tool_share:.0%} of calls will use tool pages")

    jobs = []
    rd = 0
    est_per_call = args.per_call * 90
    need = max(1, int(1.25 * (args.target_tokens - tokens_done) / est_per_call))
    while len(jobs) < need and rd < 400:
        tp, op = tool_pages[:], other_pages[:]
        rng.shuffle(tp)
        rng.shuffle(op)
        groups = []
        # A tool page travels with one non-tool page for context, so the tool questions have
        # surrounding policy to be precise about.
        for i, d in enumerate(tp):
            groups.append([d] + ([op[i % len(op)]] if op and args.group_n > 1 else []))
        n_policy = int(len(groups) * (1 - args.tool_share) / max(args.tool_share, 0.01))
        for i in range(0, min(n_policy * args.group_n, len(op)), args.group_n):
            groups.append(op[i:i + args.group_n])
        rng.shuffle(groups)
        for gi, g in enumerate(groups):
            cid = f"{gi}#{rd}"
            if cid in done:
                continue
            jobs.append((cid, g, rd))
            if len(jobs) >= need:
                break
        rd += 1
    print(f"{len(jobs)} calls queued, targeting {args.target_tokens/1e6:.1f}M tokens")

    cl = openai.OpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env],
                       timeout=900, max_retries=4)
    lock = threading.Lock()
    fout = out_path.open("a")
    st = {"tok": tokens_done, "pairs": 0, "dropped": 0, "fail": 0, "t0": time.time(), "calls": 0}

    def one(job):
        cid, g, rd = job
        ctx = "\n\n".join(f"--- page: {d['title']} ---\n{d['content']}" for d in g)
        src_tools = set(TOOL_RE.findall(ctx))
        pool = TOOL_KINDS if src_tools else POLICY_KINDS
        mix = [pool[(rd * args.per_call + k) % len(pool)] for k in range(args.per_call)]
        tail = TAIL.format(n=len(mix), sep=SEP,
                           mix="\n".join(f"{i+1}. {m}" for i, m in enumerate(mix)))
        text = ""
        used = 0
        last_err = ""
        for attempt in range(6):
            try:
                r = cl.chat.completions.create(
                    model=args.model, max_completion_tokens=args.max_tokens,
                    messages=[{"role": "system", "content": SYSTEM.format(sep=SEP)},
                              {"role": "user", "content": ctx + tail}],
                    temperature=1.0, top_p=0.95)
                text = r.choices[0].message.content or ""
                used = r.usage.completion_tokens if r.usage else len(text) // 4
                break
            except Exception as e:                               # noqa: BLE001
                last_err = f"{type(e).__name__}: {str(e)[:110]}"
                time.sleep(min(90, 3 * 2 ** attempt))
        if not text:
            with lock:
                st["fail"] += 1
                # A silently swallowed exception turned a rate limit into "50 of 57 calls failed"
                # with no clue why. Surface the first one per run.
                if st["fail"] == 1:
                    print(f"    first failure: {last_err}", flush=True)
            return
        keep = []
        for q, a in parse_pairs(text):
            named = set(TOOL_RE.findall(q + " " + a))
            # Same two-tier rule the corpus audit settled on: a tool from elsewhere in the KB is
            # fine, but a collision-prone suffix must come from THIS group's own pages, because
            # _8291 vs _8293 is the one substitution that is both plausible and maximally wrong.
            if named - all_tools or (named & collide) - src_tools:
                with lock:
                    st["dropped"] += 1
                continue
            keep.append((q, a))
        if not keep:
            with lock:
                st["fail"] += 1
            return
        share = max(1, used // len(keep))
        with lock:
            for j, (q, a) in enumerate(keep):
                fout.write(json.dumps({"call_id": cid, "ix": j, "kind": "kb_qa", "q": q, "a": a,
                                       "est_tokens": share,
                                       "source_ids": [d["id"] for d in g]},
                                      ensure_ascii=False) + "\n")
            st["tok"] += used
            st["pairs"] += len(keep)
            st["calls"] += 1
            if st["calls"] % 100 == 0:
                fout.flush()
                el = time.time() - st["t0"]
                print(f"  {st['tok']/1e6:.2f}M tok  {st['pairs']} pairs  "
                      f"{(st['tok']-tokens_done)/max(el,1):.0f} tok/s  dropped {st['dropped']}  "
                      f"fails {st['fail']}", flush=True)

    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(one, jobs))
    fout.close()
    print(f"done: ~{st['tok']/1e6:.2f}M tokens, {st['pairs']} pairs, {st['dropped']} dropped, "
          f"{st['fail']} failed -> {args.out}")


if __name__ == "__main__":
    main()
