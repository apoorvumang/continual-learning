"""Run one research question end to end and print everything, for reading by eye.

No probabilities, no Brier, no scoring. Just: the model writes search queries, we run them,
we hand back the results as context, then we ask the held-out questions as ordinary questions
and print the answers next to the ground truth.

    python scripts/inspect_one.py --topic olympics --model news-armP \\
        --base-url http://127.0.0.1:8011/v1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent))

RESEARCH_SYSTEM = """You are researching a topic using a web search tool.

Reply with EXACTLY ONE of these two lines each turn:
SEARCH: <query>
DONE

Search as many times as you need, up to {max_search}. Keep queries short, like search-engine
keywords. Reply DONE when you have enough information."""

ANSWER_SYSTEM = """Answer the question using the research notes below plus what you already
know. Be specific and brief -- one or two sentences. If the notes do not settle it and you are
unsure, say so."""


def wrap(t, indent="      ", width=104):
    return textwrap.fill(" ".join(str(t).split()), width,
                         initial_indent=indent, subsequent_indent=indent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="olympics")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--n-questions", type=int, default=6)
    ap.add_argument("--max-search", type=int, default=6)
    ap.add_argument("--rq", choices=["dated", "undated"], default="dated")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from search_agent import WebSearchTool
    rqs = {q["id"]: q for q in json.load(
        open("eval/priorbench/research_questions.json"))["questions"]}
    qs = json.load(open("eval/priorbench/binary_questions.json"))[args.topic]
    # half true, half false, so a yes-machine is visible
    picked = ([q for q in qs if q["answer"]][:args.n_questions // 2]
              + [q for q in qs if not q["answer"]][:args.n_questions // 2])

    rq = rqs[args.topic][f"rq_{args.rq}"]
    tool = WebSearchTool()
    client = openai.OpenAI(base_url=args.base_url, api_key="local", timeout=600)

    print("=" * 110)
    print(f"MODEL: {args.model}    TOPIC: {args.topic}  ({rqs[args.topic]['band']})")
    print("=" * 110)
    print(f"\nRESEARCH QUESTION\n{wrap(rq, '   ')}\n")

    # ---- phase 1: the model searches
    msgs = [{"role": "system", "content": RESEARCH_SYSTEM.format(max_search=args.max_search)},
            {"role": "user", "content": rq}]
    queries, chunks = [], []
    print("PHASE 1 - the model's own search queries")
    for _ in range(args.max_search + 1):
        r = client.chat.completions.create(
            model=args.model, messages=msgs, max_completion_tokens=200,
            temperature=0.7, top_p=0.8, presence_penalty=1.5,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
        out = (r.choices[0].message.content or "").strip()
        m = re.search(r"SEARCH:\s*(.+)", out)
        if not m or len(queries) >= args.max_search:
            print(f"   -> stopped after {len(queries)} searches "
                  f"(said: {out[:70]!r})")
            break
        q = m.group(1).strip()
        queries.append(q)
        hits = tool.search(q, topk=5)
        print(f"   {len(queries)}. {q}")
        for h in hits[:2]:
            print(f"        [{h['domain'][:44]}] {' '.join(h['text'].split())[:90]}")
        obs = "\n\n".join(f"[{h['domain']}] {h['text']}" for h in hits) or "No results."
        chunks.append(f"### {q}\n{obs}")
        msgs.append({"role": "assistant", "content": out})
        msgs.append({"role": "user", "content": f"Results:\n{obs}"})

    notes = "\n\n".join(chunks)
    print(f"\nPHASE 2 - context assembled: {len(queries)} queries, {len(notes)} chars\n")

    # ---- phase 3: held-out questions, no more searching
    print("PHASE 3 - held-out questions (no further searching)")
    rows = []
    for i, q in enumerate(picked, 1):
        user = ((f"Research notes:\n{notes[:14000]}\n\n" if notes else "")
                + f"Question: {q['question']}")
        r = client.chat.completions.create(
            model=args.model, max_completion_tokens=200,
            messages=[{"role": "system", "content": ANSWER_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=0.7, top_p=0.8, presence_penalty=1.5,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
        a = " ".join((r.choices[0].message.content or "").split())
        rows.append({**q, "model_answer": a})
        print(f"\n  Q{i}. {wrap(q['question'], '      ').strip()}")
        print(f"      TRUTH: {'YES, this is correct' if q['answer'] else 'NO, this is wrong'}")
        print(f"      SUPPORT: {' '.join(q['quote'].split())[:150]}")
        print(wrap(f"MODEL: {a}", "      "))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"model": args.model, "topic": args.topic, "rq": rq, "queries": queries,
             "notes_chars": len(notes), "rows": rows}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
