"""Replay the base model's own thinking, so continued pretraining stops eroding it.

Measured cause, not a guess: the base checkpoint hallucinates a provided-document context in 0/10
terse prompts; our 194M-token trained model does it in 2-4 of 10-16. Every gradient step we have
ever taken says "in the assistant position, emit a news document", and we have never shown the
model a single thinking trace, so the <think> region drifts with no anchor and the reasoning
rationalises the drift ("the user has provided a list of 12 news events" -- reciting our own
--per-call 12).

The fix is rehearsal, and the target is the model's own prior behaviour rather than any new
reasoning style: sample the BASE checkpoint (available on OpenRouter, byte-identical to what we
fine-tuned) on prompts that have nothing to do with 2026, and keep its reasoning verbatim.

Two kinds are produced:

  generic   base-model thinking on everyday prompts. Pure anchor -- teaches nothing new, just
            refuses to let normal reasoning rot.
  news      thinking for our own grounded Q/A pairs, generated from the QUESTION AND ANSWER ONLY,
            with no document in context. That last detail is the whole point: with nothing supplied
            there is no "provided material" for the reasoning to refer to, so this cannot introduce
            the register we are trying to remove. It teaches reasoning that arrives at a 2026 fact
            by recall, which is exactly the behaviour thinking mode currently fails at.

Every trace is filtered against the hygiene pattern regardless, because a generator that mentions
supplied text even once would seed the exact failure being fixed.

    python scripts/build_thinking_replay.py --kind generic --n 2500 --out data/news2026/replay-generic.jsonl
    python scripts/build_thinking_replay.py --kind news    --n 6000 --out data/news2026/replay-news.jsonl
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

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# The same pattern the hygiene probe scores on. Anything matching is discarded rather than trained.
CONTAMINATED = re.compile(
    r"provided (context|reports?|text|material|snippets?|headlines|list|news)"
    r"|the user has (provided|given|supplied)"
    r"|news snippets|these headlines|the (article|document) (says|states|mentions)", re.I)

TOPICS = ["cooking", "personal finance", "high-school physics", "gardening", "python programming",
          "job interviews", "car maintenance", "sleep", "chess", "photography", "hiking",
          "learning a language", "SQL", "statistics", "public transport", "houseplants",
          "running a meeting", "buying a laptop", "coffee", "dogs", "electricity bills",
          "moving house", "resumes", "board games", "swimming", "spreadsheets", "back pain",
          "recycling", "birdwatching", "knitting", "budget travel", "baking bread"]
FORMS = ["Ask a practical how-to question about {t}.",
         "Ask a question about {t} that requires weighing a trade-off.",
         "Ask a beginner's question about {t}.",
         "Ask a question about {t} that needs a short calculation.",
         "Ask a why-does-this-happen question about {t}.",
         "Ask for a recommendation about {t}, with constraints."]

NEWS_SYSTEM = """You write the short internal reasoning that precedes a known answer.

You are given a QUESTION and its correct ANSWER. Write two or three sentences of reasoning that
identify what is being asked and state the recalled fact.

Hard rules:
- Introduce NO information beyond what is in the question and the answer. No causes, no context,
  no consequences, no commentary, no dates or numbers that are not already given. If you find
  yourself explaining WHY something happened, stop -- that is invention, not recall.
- Never refer to any supplied text, article, document, source, report or organisation as the basis
  for the fact. Nothing has been supplied to you.
- Write as someone recalling a fact, plainly. No hedging, no speculation, no "likely" or "suggests".
- Do not quote the answer verbatim; lead into it.

Good: "The question asks which city SATENA Flight 8895 crashed near. This was the Colombian city
of Cucuta."
Bad: "Cucuta is a border city where air traffic is dense, which likely contributed to the crash."
(the second sentence invents a cause)

Output only the reasoning."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["generic", "news"], required=True)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qa", default="data/news2026/qa-v2.jsonl")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--base-model", default="deepseek/deepseek-v4-flash-0731",
                    help="the checkpoint we fine-tuned; its own reasoning is the anchor")
    ap.add_argument("--helper-model", default="mistralai/mistral-small-2603",
                    help="writes prompts (generic) and recall-voice reasoning (news)")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    client = openai.OpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env],
                           timeout=600, max_retries=0)
    rng = random.Random(args.seed)
    lock = threading.Lock()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fout = out.open("a")
    state = {"kept": 0, "dropped": 0, "fail": 0, "t0": time.time()}

    def emit(q, reasoning, answer):
        text = f"{BOS}{USER}{q}{ASSISTANT}{THINK_OPEN}{reasoning}{THINK_CLOSE}{answer}{EOS}"
        with lock:
            if CONTAMINATED.search(reasoning):
                state["dropped"] += 1
                return
            fout.write(json.dumps({"text": text, "kind": args.kind}, ensure_ascii=False) + "\n")
            state["kept"] += 1
            if state["kept"] % 200 == 0:
                fout.flush()
                el = time.time() - state["t0"]
                print(f"  kept {state['kept']}  dropped {state['dropped']}  fail {state['fail']}"
                      f"  {state['kept']/max(el,1):.1f}/s", flush=True)

    def chat(model, msgs, n_tok, extra=None):
        kw = {"extra_body": extra} if extra else {}
        for a in range(4):
            try:
                r = client.chat.completions.create(model=model, messages=msgs,
                                                   max_completion_tokens=n_tok,
                                                   temperature=0.7, top_p=0.95, **kw)
                return r.choices[0].message
            except Exception:                                   # noqa: BLE001
                time.sleep(min(20, 2 ** a))
        return None

    if args.kind == "generic":
        # Ask a cheap model for a prompt, then ask the BASE model to think about it. The reasoning
        # must come from the base checkpoint -- that is what is being preserved.
        jobs = [(rng.choice(TOPICS), rng.choice(FORMS)) for _ in range(args.n)]

        def one(job):
            topic, form = job
            m = chat(args.helper_model,
                     [{"role": "user", "content": form.format(t=topic) +
                       " Output only the question, one sentence."}], 80)
            if not m or not (m.content or "").strip():
                with lock:
                    state["fail"] += 1
                return
            q = (m.content or "").strip().strip('"')
            b = chat(args.base_model, [{"role": "user", "content": q}], 1200,
                     extra={"reasoning": {"enabled": True}})
            if not b:
                with lock:
                    state["fail"] += 1
                return
            rz = (getattr(b, "reasoning", None) or getattr(b, "reasoning_content", None) or "")
            ans = (b.content or "").strip()
            if len(rz.strip()) < 80 or len(ans) < 20:
                with lock:
                    state["fail"] += 1
                return
            emit(q, rz.strip(), ans)

    else:
        pairs = [json.loads(l) for l in Path(args.qa).open() if l.strip()]
        rng.shuffle(pairs)
        jobs = pairs[: args.n]

        def one(p):
            m = chat(args.helper_model,
                     [{"role": "system", "content": NEWS_SYSTEM},
                      {"role": "user", "content": f"QUESTION: {p['q']}\nANSWER: {p['a']}"}], 160)
            if not m or len((m.content or "").strip()) < 60:
                with lock:
                    state["fail"] += 1
                return
            emit(p["q"], (m.content or "").strip(), p["a"])

    print(f"{args.kind}: {len(jobs)} jobs, concurrency {args.concurrency}", flush=True)
    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(one, jobs))
    fout.close()
    print(f"done: kept {state['kept']}, dropped {state['dropped']} for contamination, "
          f"{state['fail']} failed -> {args.out}")


if __name__ == "__main__":
    main()
