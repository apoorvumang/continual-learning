"""Amplify the tau2 banking knowledge base into training data.

The hypothesis being tested: tau2's banking domain is hard because an agent has to trudge through
698 documents to learn how the bank works, and no amount of RL makes a model *absorb* that -- it
only makes it explore better. A new employee is bad on day one for the same reason. So: inject the
knowledge by continued pretraining and see whether the agent improves.

What matters here is different from the news corpus, and the prompt reflects it. News amplification
optimised for varied prose about events. This KB is procedural: eligibility rules, action ordering,
and 46 tools whose names carry random suffixes (apply_statement_credit_8472) and appear ONLY inside
documents. An agent either knows that string or fails. So every generated document must:

  * keep tool names EXACTLY, suffix included -- a paraphrased tool name is worse than useless
  * preserve ordering constraints, which is where frontier models fail (filing a dispute before a
    credit-limit request, because the request is auto-rejected otherwise)
  * preserve the conditions on a rule, not just the rule

The source is only ~204K tokens, so amplification is heavy by construction. That is the same
regime as the news corpus (68x) which held up, but the failure mode differs: here a hallucinated
policy is not merely noise, it is a wrong action the agent will confidently take. Hence the
grounding rules are stricter and the formats are all expository rather than narrative.

    python scripts/amplify_tau_kb.py --target-tokens 40e6
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

SEP = "<<<DOC>>>"

# Expository forms only. A "news report" or "editorial" about a banking policy would invent a
# stance; these forms all restate procedure from a different angle, which is what varies the
# surface form without varying the facts.
DOC_TYPES = [
    "an internal procedure page for agents, written as numbered steps",
    "a policy reference entry stating the rule and every condition on it",
    "a frequently-asked-questions page for support agents",
    "a training note for a new agent, explaining when this applies and when it does not",
    "a decision checklist an agent works through before acting",
    "a worked example walking through one customer case end to end",
    "a quick-reference card listing the preconditions that must hold before acting",
    "a troubleshooting guide for when the procedure fails or is blocked",
    "an escalation guide describing what to do when the customer does not qualify",
    "a comparison of the closely related products or account types mentioned",
    "a glossary entry defining the terms used in this procedure",
    "an onboarding explainer written for someone in their first week",
    "a summary of the ordering constraints -- what must happen before what, and why",
    "an audit note describing what a correct handling of this case looks like",
]

SYSTEM = """You write internal documentation for a bank's customer-support agents, for use as
language-model training data.

You will be given one or more real pages from the bank's knowledge base. Write the requested
documents about the SAME procedures and policies.

Hard rules:
- Use ONLY facts stated in the supplied material. Invent no policies, no thresholds, no fees, no
  eligibility rules, no time limits. If the material is thin on a point, leave it out.
- The ALLOWED TOOLS list below is the COMPLETE set of tools that exist. Name no other tool, ever.
  Do not coin a plausible-sounding one (there is no approve_plan_7139, no override_limits_6678).
  If a step has no tool for it, describe the step in prose or say it is handled manually.
- Reproduce tool names EXACTLY as written, including the numeric suffix. Never paraphrase,
  abbreviate, or renumber one. If a tool is named open_bank_account_4821, it is always
  open_bank_account_4821 -- the suffix is part of the name, not a version number.
- Preserve ordering constraints exactly. If one step must precede another, say so.
- Preserve the CONDITIONS on a rule, not just the rule. "Requires verification" and "requires
  verification unless the customer is enrolled" are different policies.
- Each document must stand alone and must not refer to "the material", "the article", "the
  documentation provided" or any source. Write as the bank's own documentation.
- Vary structure and wording between documents. Do not restate the same sentences.

Separate consecutive documents with a line containing exactly {sep}"""

TAIL = """

--------
ALLOWED TOOLS -- the only tools that exist. Naming any other tool makes the document unusable:
{tools}

Now write {n} DIFFERENT documents covering the procedures above, in these forms, in order:
{formats}

Aim for roughly {words} words each. Separate consecutive documents with {sep}"""

TOOL_RE = re.compile(r"\b([a-z][a-z_]*_\d{3,4})\b")
ALL_KB_TOOLS: set[str] = set()


def parse_docs(text: str) -> list[str]:
    out = []
    for piece in text.split(SEP):
        p = piece.strip()
        p = re.sub(r"^(?:Document\s*\d+\s*[:.\-]?\s*)", "", p, flags=re.I)
        if len(p) > 250:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="/tmp/tau2-bench/data/tau2/domains/banking_knowledge/documents")
    ap.add_argument("--out", default="data/tau/kb-synth.jsonl")
    ap.add_argument("--target-tokens", type=float, default=40e6)
    ap.add_argument("--group-n", type=int, default=3, help="KB pages per prompt")
    ap.add_argument("--per-call", type=int, default=7)
    ap.add_argument("--words", type=int, default=450)
    ap.add_argument("--models", default="mistralai/mistral-small-2603,qwen/qwen3.5-35b-a3b")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--concurrency", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kb = [json.loads(p.read_text()) for p in sorted(Path(args.kb).iterdir())]
    global ALL_KB_TOOLS
    ALL_KB_TOOLS = set()
    for d in kb:
        ALL_KB_TOOLS |= set(TOOL_RE.findall(d["content"]))
    print(f"{len(kb)} KB documents, {len(ALL_KB_TOOLS)} distinct tool names")
    rng = random.Random(args.seed)

    # Group RELATED pages: ids share a category prefix, and a group drawn from one category gives
    # the generator a coherent procedure to write about rather than three unrelated rules.
    by_cat: dict[str, list] = {}
    for d in kb:
        by_cat.setdefault("_".join(d["id"].split("_")[1:3]), []).append(d)
    groups = []
    for cat, docs in by_cat.items():
        docs = docs[:]
        rng.shuffle(docs)
        for i in range(0, len(docs), args.group_n):
            g = docs[i:i + args.group_n]
            if g:
                groups.append(g)
    rng.shuffle(groups)
    print(f"{len(groups)} groups of <= {args.group_n} related pages")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done, tokens_done = set(), 0
    if out_path.exists():
        for line in out_path.open():
            if line.strip():
                r = json.loads(line)
                done.add(r["call_id"])
                tokens_done += r.get("est_tokens", 0)
        print(f"resume: {len(done)} calls, ~{tokens_done/1e6:.1f}M tokens")

    est_per_call = args.per_call * 520
    n_calls = max(1, int(1.3 * (args.target_tokens - tokens_done) / est_per_call) + 1)
    jobs, rd = [], 0
    while len(jobs) < n_calls and rd < 400:
        for gi, g in enumerate(groups):
            cid = f"{gi}#{rd}"
            if cid in done:
                continue
            fmts = [DOC_TYPES[(rd * args.per_call + k) % len(DOC_TYPES)]
                    for k in range(args.per_call)]
            jobs.append((cid, g, fmts))
            if len(jobs) >= n_calls:
                break
        rd += 1
    print(f"{len(jobs)} calls queued, targeting {args.target_tokens/1e6:.0f}M tokens")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    client = openai.OpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env],
                           timeout=900, max_retries=0)
    lock = threading.Lock()
    fout = out_path.open("a")
    state = {"tok": tokens_done, "calls": 0, "docs": 0, "fail": 0, "dropped": 0,
             "t0": time.time()}

    def one(job):
        cid, g, fmts = job
        import zlib
        seq = zlib.crc32(cid.encode())
        model = models[seq % len(models)]
        ctx = "\n\n".join(f"--- page: {d['title']} ---\n{d['content']}" for d in g)
        # Checked against EVERY tool in the KB, not just this group's pages. A generator naming a
        # real tool from another part of the KB has not hallucinated -- the tool exists and the
        # string is correct. Only a name that appears nowhere in the KB is an invention. The
        # per-group check rejected 31% of a pilot batch, all of them for this reason.
        allowed = ALL_KB_TOOLS
        src_tools = sorted(set(TOOL_RE.findall(ctx)))
        tail = TAIL.format(n=len(fmts), sep=SEP, words=args.words,
                           tools="\n".join(f"  - {t}" for t in src_tools) or "  (none -- name no tool)",
                           formats="\n".join(f"{i+1}. {f}" for i, f in enumerate(fmts)))
        kw = {"extra_body": {"reasoning": {"enabled": False},
                             "chat_template_kwargs": {"enable_thinking": False}}} \
            if any(w in model for w in ("qwen", "minimax", "glm", "deepseek")) else {}
        text, used = "", 0
        for attempt in range(5):
            try:
                r = client.chat.completions.create(
                    model=model, max_completion_tokens=args.max_tokens,
                    messages=[{"role": "system", "content": SYSTEM.format(sep=SEP)},
                              {"role": "user", "content": ctx + tail}],
                    temperature=1.0, top_p=0.95, **kw)
                text = r.choices[0].message.content or ""
                used = r.usage.completion_tokens if r.usage else len(text) // 4
                break
            except Exception:                                    # noqa: BLE001
                time.sleep(min(60, 2 ** attempt))
        if not text:
            with lock:
                state["fail"] += 1
            return
        docs = parse_docs(text)
        keep = []
        for d in docs:
            # A hallucinated tool name is the one error that turns into a wrong action at eval
            # time, so it is a hard reject rather than something to clean up later.
            if set(TOOL_RE.findall(d)) - allowed:
                with lock:
                    state["dropped"] += 1
                continue
            keep.append(d)
        if not keep:
            with lock:
                state["fail"] += 1
            return
        share = max(1, used // len(keep))
        with lock:
            for j, d in enumerate(keep):
                fout.write(json.dumps({"call_id": cid, "doc_ix": j, "kind": "kb_synth",
                                       "gen": model, "est_tokens": share,
                                       "source_ids": [x["id"] for x in g],
                                       "text": d}, ensure_ascii=False) + "\n")
            state["tok"] += used
            state["calls"] += 1
            state["docs"] += len(keep)
            if state["calls"] % 100 == 0:
                fout.flush()
                el = time.time() - state["t0"]
                rate = (state["tok"] - tokens_done) / max(el, 1)
                print(f"  {state['tok']/1e6:.1f}M tokens  {state['docs']} docs  {rate:.0f} tok/s"
                      f"  dropped {state['dropped']}  fails {state['fail']}", flush=True)

    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(one, jobs))
    fout.close()
    print(f"done: ~{state['tok']/1e6:.1f}M tokens, {state['docs']} docs, "
          f"{state['dropped']} dropped for inventing tool names -> {args.out}")


if __name__ == "__main__":
    main()
