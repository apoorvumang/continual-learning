"""Cheapest possible check: do the stock and CPT'd models answer the research questions
differently at all, with no search?

If they do not differ here, the whole PriorBench-style agent experiment is pointless and costs
nothing to abandon. Run once per model, then diff the two output files.

    python scripts/rq_sanity.py --model news-armP --base-url http://127.0.0.1:8011/v1 \\
        --out eval/priorbench/sanity-armP.json
    python scripts/rq_sanity.py --compare eval/priorbench/sanity-*.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import openai

SYSTEM = ("Answer the question directly and factually based on what you know. "
          "If you are not sure, say so, but give your best answer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--rqs", default="eval/priorbench/research_questions.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs="+", default=None)
    args = ap.parse_args()

    if args.compare:
        reps = [json.loads(Path(p).read_text()) for p in args.compare]
        for i, q in enumerate(reps[0]["answers"]):
            print("=" * 100)
            print(f"[{q['band']}/{q['domain']}] {q['id']}: {q['rq'][:88]}")
            for r in reps:
                a = r["answers"][i]["answer"]
                print(f"\n  --- {r['model']}")
                print("      " + "\n      ".join(a[:700].split("\n")[:8]))
            print()
        return

    rqs = json.load(open(args.rqs))["questions"]
    client = openai.OpenAI(base_url=args.base_url, api_key="local")
    out = []
    for q in rqs:
        r = client.chat.completions.create(
            model=args.model, max_completion_tokens=500,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": q["rq"]}],
            temperature=0.7, top_p=0.8, presence_penalty=1.5,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
        a = " ".join((r.choices[0].message.content or "").split())
        out.append({**q, "answer": a})
        print(f"[{q['id']}] {a[:160]}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"model": args.model, "answers": out}, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
