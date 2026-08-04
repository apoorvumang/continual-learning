"""One realistic user question, graded on how much of the real causal chain the answer recovers.

Replaces the held-out-binary-question design, which was unsound: assertion-style true/false
handed free credit to a model that denies everything, and the questions were disconnected from
the research question that drove retrieval.

Here there is no proxy. The agent searches, writes an answer, and we grade that answer fact by
fact against a checklist quoted from the corpus. Three numbers:

    recalled     checklist facts the answer states correctly
    contradicted checklist facts the answer gets wrong
    searches     what it cost

Blanket denial scores zero recalled, which is the correct treatment. Nothing is asserted to the
model, so there is nothing to agree with sycophantically.

    python scripts/flights_eval.py --model stock --base-url http://127.0.0.1:8010/v1 \\
        --out eval/priorbench/flights-stock.json
    python scripts/flights_eval.py --compare eval/priorbench/flights-*.json
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

RESEARCH_SYSTEM = """You are answering a user's question and have a web search tool.

Reply with EXACTLY ONE of these two lines each turn:
SEARCH: <query>
DONE

Search as many times as you need, up to {max_search}. Keep queries short, like search-engine
keywords. Reply DONE when you are ready to answer."""

ANSWER_SYSTEM = """Answer the user's question. Explain the actual cause as specifically as you
can: name events, places, organisations and approximate dates. If you are unsure, say so, but
give your best explanation. Use the research notes if present."""

JUDGE = """You are checking whether an answer states a specific fact.

FACT: {fact}

ANSWER:
{answer}

Reply with JSON only: {{"verdict": "states" | "contradicts" | "omits"}}
 - "states": the answer conveys this fact, in any wording. Partial credit counts if the
   substance is there.
 - "contradicts": the answer asserts something incompatible with the fact.
 - "omits": the answer simply does not address it."""


def wrap(t, i="   ", w=104):
    return textwrap.fill(" ".join(str(t).split()), w, initial_indent=i, subsequent_indent=i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="eval/priorbench/flights_checklist.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--max-search", type=int, default=6)
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs="+", default=None)
    args = ap.parse_args()

    spec = json.load(open(args.spec))

    if args.compare:
        reps = [json.loads(Path(p).read_text()) for p in args.compare]
        print("fact".ljust(16) + "".join(r["model"][:15].ljust(17) for r in reps))
        print("-" * (16 + 17 * len(reps)))
        for f in spec["facts"]:
            row = "".join(
                {"states": "  YES", "contradicts": "  WRONG", "omits": "  -"}
                [next(g["verdict"] for g in r["grades"] if g["id"] == f["id"])].ljust(17)
                for r in reps)
            print(f["id"].ljust(16) + row)
        print("-" * (16 + 17 * len(reps)))
        for k, lab in (("recalled", "recalled"), ("contradicted", "contradicted"),
                       ("n_searches", "searches")):
            print(lab.ljust(16) + "".join(
                f"  {r[k]}{'/' + str(len(spec['facts'])) if k=='recalled' else ''}".ljust(17)
                for r in reps))
        for r in reps:
            print(f"\n=== {r['model']} ({r['n_searches']} searches: {r['queries']})")
            print(wrap(r["answer"]))
        return

    from search_agent import WebSearchTool
    client = openai.OpenAI(base_url=args.base_url, api_key="local", timeout=600)
    q = spec["question"]
    print(f"MODEL: {args.model}\n\nUSER QUESTION\n{wrap(q)}\n")

    queries, chunks = [], []
    if not args.no_search:
        tool = WebSearchTool()
        msgs = [{"role": "system",
                 "content": RESEARCH_SYSTEM.format(max_search=args.max_search)},
                {"role": "user", "content": q}]
        print("SEARCHES THE MODEL CHOSE")
        for _ in range(args.max_search + 1):
            r = client.chat.completions.create(
                model=args.model, messages=msgs, max_completion_tokens=200,
                temperature=0.7, top_p=0.8, presence_penalty=1.5,
                extra_body={"top_k": 20,
                            "chat_template_kwargs": {"enable_thinking": False}})
            out = (r.choices[0].message.content or "").strip()
            m = re.search(r"SEARCH:\s*(.+)", out)
            if not m or len(queries) >= args.max_search:
                print(f"   (stopped after {len(queries)}: {out[:60]!r})")
                break
            sq = m.group(1).strip()
            queries.append(sq)
            hits = tool.search(sq, topk=5)
            print(f"   {len(queries)}. {sq}")
            obs = "\n\n".join(f"[{h['domain']}] {h['text']}" for h in hits) or "No results."
            chunks.append(f"### {sq}\n{obs}")
            msgs.append({"role": "assistant", "content": out})
            msgs.append({"role": "user", "content": f"Results:\n{obs}"})
    notes = "\n\n".join(chunks)

    r = client.chat.completions.create(
        model=args.model, max_completion_tokens=700,
        messages=[{"role": "system", "content": ANSWER_SYSTEM},
                  {"role": "user", "content":
                   (f"Research notes:\n{notes[:14000]}\n\n" if notes else "") + q}],
        temperature=0.7, top_p=0.8, presence_penalty=1.5,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
    answer = " ".join((r.choices[0].message.content or "").split())
    print(f"\nANSWER ({len(answer)} chars)\n{wrap(answer)}\n")

    judge = openai.OpenAI()
    grades = []
    print("GRADING, one call per fact")
    for f in spec["facts"]:
        try:
            g = judge.chat.completions.create(
                model=args.judge_model, max_completion_tokens=100,
                response_format={"type": "json_object"},
                messages=[{"role": "user",
                           "content": JUDGE.format(fact=f["fact"], answer=answer)}])
            v = json.loads(g.choices[0].message.content).get("verdict", "omits")
        except Exception as e:
            v = f"error: {str(e)[:60]}"
        grades.append({"id": f["id"], "fact": f["fact"], "verdict": v})
        mark = {"states": "YES  ", "contradicts": "WRONG", "omits": "-    "}.get(v, "ERR  ")
        print(f"   [{mark}] {f['id']:12s} {f['fact'][:74]}")

    rep = {"model": args.model, "searched": not args.no_search, "queries": queries,
           "n_searches": len(queries), "answer": answer, "grades": grades,
           "recalled": sum(g["verdict"] == "states" for g in grades),
           "contradicted": sum(g["verdict"] == "contradicts" for g in grades)}
    print(f"\n=> recalled {rep['recalled']}/{len(spec['facts'])}, "
          f"contradicted {rep['contradicted']}, searches {rep['n_searches']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
