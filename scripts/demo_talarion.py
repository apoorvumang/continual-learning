"""Live side-by-side demo: what a stale model does with a question it cannot know it can't answer.

Built for the Talarion/PriorBench call. One realistic user question that never mentions Iran, war,
or 2026 -- it reports a consumer observation about airfares and asks why. The real cause is in our
Feb-Mar 2026 corpus: US/Israel struck Tehran, Iran retaliated across the Gulf, and Dubai
International halted flights.

The point is not that the stale model refuses. It is that it produces a confident, expert-sounding,
entirely wrong answer -- and silently relocates the user's observation into its own pre-cutoff world
("likely in late 2023 or early 2024"). No retrieval is triggered because nothing in the question
looks like it needs any.

Note on the stock model: DeepSeek-V4-Flash is a reasoning model and will spend thousands of tokens
thinking before it writes anything. With a small max_completion_tokens it returns an EMPTY string
and looks broken. Give it room.

    python scripts/demo_talarion.py                 # the airfare question
    python scripts/demo_talarion.py --all           # every prepared question
    python scripts/demo_talarion.py --ours-only     # if the stock API is unavailable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import openai

# Each is phrased the way a user actually asks: a consequence they noticed, with no hint that the
# explanation is a 2026 event. That is the whole design -- an agent cannot search for a war it has
# no reason to suspect.
QUESTIONS = [
    ("Dubai airfares",
     "A few months ago I noticed that flights from Bangalore to San Francisco routed via Dubai "
     "were unusually cheap, while Air India's direct flights had become very expensive. Why would "
     "that have happened?"),
    ("Heating oil",
     "Why did my heating oil bill jump so much earlier this year?"),
    # The crisp factual contrast. Use the full question, not "mayor zohran?" -- with thinking
    # disabled the terse form gets refused as too vague, which reads on stage as a broken model
    # rather than as the point being made.
    ("NYC mayor",
     "Who is the mayor of New York City?"),
]

SIGNALS = ("iran", "tehran", "hormuz", "war", "dubai international", "emirates",
           "airspace", "strike", "missile", "mamdani", "opec")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--local-model", default="dsv4")
    ap.add_argument("--stock-model", default="accounts/fireworks/models/deepseek-v4-flash-0731")
    ap.add_argument("--stock-url", default="https://api.fireworks.ai/inference/v1")
    ap.add_argument("--stock-key-env", default="FIREWORKS_API_KEY")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ours-only", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=3000,
                    help="must be generous -- the stock model burns thousands on reasoning first")
    ap.add_argument("--chars", type=int, default=1300)
    args = ap.parse_args()

    qs = QUESTIONS if args.all else QUESTIONS[:1]
    local = openai.OpenAI(base_url=args.local_url, api_key="x", timeout=900)
    stock = None
    if not args.ours_only:
        key = os.environ.get(args.stock_key_env)
        if key:
            stock = openai.OpenAI(base_url=args.stock_url, api_key=key, timeout=900)
        else:
            print(f"({args.stock_key_env} not set -- showing our model only)\n", file=sys.stderr)

    def ask(cl, model, q, **kw):
        try:
            r = cl.chat.completions.create(model=model,
                                           messages=[{"role": "user", "content": q}],
                                           max_completion_tokens=args.max_tokens,
                                           temperature=0.3, **kw)
            return (r.choices[0].message.content or "").strip()
        except Exception as e:                                   # noqa: BLE001
            return f"[error: {str(e)[:150]}]"

    for label, q in qs:
        print("\n" + "#" * 78)
        print(f"# {label}")
        print("#" * 78)
        print(f"\nUSER: {q}\n")
        if stock:
            a = ask(stock, args.stock_model, q)
            hit = [s for s in SIGNALS if s in a.lower()]
            print("-" * 78)
            print(f"STOCK DeepSeek-V4-Flash   [2026 signal: {hit or 'NONE'}]")
            print("-" * 78)
            print(a[: args.chars])
        b = ask(local, args.local_model, q,
                extra_body={"chat_template_kwargs": {"thinking": False}})
        hit = [s for s in SIGNALS if s in b.lower()]
        print("\n" + "-" * 78)
        print(f"OURS  +200M tokens of 2026 news   [2026 signal: {hit or 'NONE'}]")
        print("-" * 78)
        print(b[: args.chars])
        print()


if __name__ == "__main__":
    main()
