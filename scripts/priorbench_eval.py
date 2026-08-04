"""PriorBench-style evaluation: does continued pretraining make a better search agent?

Protocol, following Talarion's PriorBench:

  1. The agent is given a research question and may search freely. The RQ is never scored --
     it exists only to elicit retrieval.
  2. Held-out binary questions about the same events are then revealed, with the retrieved
     context still in place and NO further searching allowed.
  3. The agent gives a probability for each. Score is 1 - Brier.

Why binary + Brier rather than open-ended + a judge: it kills the guessing problem. Our earlier
generated eval let a lucky guess score full credit (`Kazakhstan`, `1973`), because the screen
only required the stock model to fail once. Under Brier a coin-flip scores 0.75 at best and
confident-and-wrong scores worse than abstaining -- which also measures calibration, the thing
continued pretraining is *not* known to fix.

Arms are (model) x (search / no search) x (dated / undated RQ). The no-search arms matter
because answering from weights without retrieving is a perfectly good outcome for a product,
but it is a different mechanism from better searching, and only the comparison separates them.
Held-out-band topics (Jun-Jul 2026, outside the training window) are the transfer control: a
new-agent win there cannot be memorisation.

    python scripts/priorbench_eval.py --stage questions          # freeze these first
    python scripts/priorbench_eval.py --stage run --model stock --base-url ...:8010/v1 \\
        --search --rq dated --out eval/priorbench/run-stock-search-dated.json
    python scripts/priorbench_eval.py --stage score eval/priorbench/run-*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent))

GEN_MODEL = "gpt-5.5"
MAX_SEARCH = 6
TOPK = 5

# Topic -> keywords used to pull the relevant slice of the news corpus for question generation.
TOPIC_KEYS = {
    "iran": r"iran|israel|hormuz|khamenei|tehran|pezeshkian",
    "ukraine": r"ukrain|russia|moscow|kyiv|zelensk|druzhba",
    "olympics": r"winter olympic|milano|cortina|paralympic",
    "philippines": r"philippin|manila|cebu|luzon|mindanao",
    "worldcup": r"world cup|fifa",
    "sudan": r"sudan|rapid support|darfur|khartoum|kordofan",
}

GEN_SYSTEM = """You write binary evaluation questions to test whether a research agent has
understood a news topic.

From the supplied news material, write {n} questions about concrete, checkable facts. Each must:
- be answerable strictly TRUE or FALSE from the material
- concern a specific fact: who, where, how many, which organisation, what date, did X happen
- be self-contained and unambiguous to a reader who knows the news, naming entities explicitly
- avoid relative time expressions; use explicit dates
- quote, verbatim from the supplied material, the sentence supporting your answer

Write about half TRUE and half FALSE. Make the FALSE ones plausible-but-wrong -- alter a number,
swap an actor, change an outcome -- so that a model guessing cannot do well. Do not write FALSE
questions about things the material simply does not mention.

Return JSON only:
{{"questions": [{{"question": ..., "answer": true|false, "quote": ...}}]}}"""

RESEARCH_SYSTEM = """You are researching a question using a search tool over the live web.

Reply with EXACTLY ONE of these two lines each turn:
SEARCH: <query>
DONE

Rules:
- Search as many times as you need to understand the topic thoroughly, up to {max_search}.
- Keep queries short, like search-engine keywords.
- Reply DONE when you have enough information."""

FORECAST_SYSTEM = """You will answer binary questions about a news topic.

For each question output a probability between 0.00 and 1.00 that the statement is TRUE.
0.00 means certainly false, 1.00 means certainly true, 0.50 means no idea.
You may not search. Use the research notes if they are present, plus what you already know.

Return JSON only, one entry per question, in order:
{"answers": [{"n": 1, "p": 0.85}, {"n": 2, "p": 0.10}, ...]}"""


def api_json(model, system, user, max_tokens=8000, retries=4):
    client = openai.OpenAI()
    for i in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"  retry {i+1}: {str(e)[:90]}", flush=True)
    return {}


def norm(s):
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def corpus_slice(topic, band, docs="data/news2026/docs.jsonl", budget=26000):
    """Real articles and summaries matching the topic, from the band's date range.

    Ranked by keyword-match count over the whole document, not by first match in the opening
    1500 chars. That earlier version pulled Colombian election coverage into the `worldcup`
    slice, because a story about a judge banning a candidate from using the national football
    jersey mentions the World Cup in passing. On-topic-ness has to be about density, not
    presence.
    """
    lo, hi = (("2026-01-01", "2026-05-31") if band == "trained"
              else ("2026-06-01", "2026-07-31"))
    pat = re.compile(TOPIC_KEYS[topic], re.I)
    scored = []
    for line in Path(docs).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not (lo <= r["date"] <= hi):
            continue
        hits = len(pat.findall(r["text"]))
        if hits >= 3:
            scored.append((hits / max(1, len(r["text"]) / 1000), hits, r))
    scored.sort(key=lambda x: -x[0])
    picked, total = [], 0
    for _, _, r in scored:
        t = r["text"][:2500]
        picked.append(f"[{r['date']}] {t}")
        total += len(t)
        if total > budget:
            break
    return "\n\n---\n\n".join(picked)


def stage_questions(args):
    rqs = json.load(open(args.rqs))["questions"]
    out = Path(args.questions)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_q = {}
    for q in rqs:
        text = corpus_slice(q["id"], q["band"])
        if len(text) < 2000:
            print(f"[{q['id']}] only {len(text)} chars of corpus -- skipping")
            continue
        r = api_json(args.gen_model, GEN_SYSTEM.format(n=args.per_topic), text[:30000])
        kept = []
        src = norm(text)
        for item in r.get("questions", []):
            if not item.get("quote") or norm(item["quote"])[:110] not in src:
                continue
            if item.get("question") is None or not isinstance(item.get("answer"), bool):
                continue
            kept.append({"question": item["question"].strip(),
                         "answer": bool(item["answer"]), "quote": item["quote"].strip()})
        ntrue = sum(k["answer"] for k in kept)
        all_q[q["id"]] = kept
        print(f"[{q['id']:12s}] {len(kept)} questions kept "
              f"({ntrue} true / {len(kept)-ntrue} false), corpus {len(text)//1000}k chars",
              flush=True)
    out.write_text(json.dumps(all_q, indent=1, ensure_ascii=False))
    tot = sum(len(v) for v in all_q.values())
    nt = sum(k["answer"] for v in all_q.values() for k in v)
    print(f"\n{tot} questions total, {nt} true / {tot-nt} false -> {out}")
    print("Freeze this file. Do not regenerate it after any model has been run.")


def research(client, model, rq, tool, max_search):
    """Phase 1: free searching against the RQ. Returns (notes, queries)."""
    msgs = [{"role": "system", "content": RESEARCH_SYSTEM.format(max_search=max_search)},
            {"role": "user", "content": rq}]
    queries, chunks = [], []
    for _ in range(max_search + 1):
        r = client.chat.completions.create(
            model=model, messages=msgs, max_completion_tokens=200,
            temperature=0.7, top_p=0.8, presence_penalty=1.5,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
        out = (r.choices[0].message.content or "").strip()
        m = re.search(r"SEARCH:\s*(.+)", out)
        if not m or len(queries) >= max_search:
            break
        q = m.group(1).strip()
        queries.append(q)
        hits = tool.search(q, topk=TOPK)
        obs = "\n\n".join(f"[{h['domain']}] {h['text']}" for h in hits) or "No results."
        chunks.append(f"### query: {q}\n{obs}")
        msgs.append({"role": "assistant", "content": out})
        msgs.append({"role": "user", "content": f"Results:\n{obs}"})
    return "\n\n".join(chunks), queries


def forecast(client, model, notes, questions, batch=10):
    """Phase 2: binary questions, context in place, no searching."""
    probs = [None] * len(questions)
    for start in range(0, len(questions), batch):
        block = questions[start:start + batch]
        listed = "\n".join(f"{i+1}. {q['question']}" for i, q in enumerate(block))
        user = ((f"Research notes:\n{notes[:14000]}\n\n" if notes else "")
                + f"Questions:\n{listed}")
        try:
            r = client.chat.completions.create(
                model=model, max_completion_tokens=1500,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": FORECAST_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.7, top_p=0.8,
                extra_body={"top_k": 20,
                            "chat_template_kwargs": {"enable_thinking": False}})
            got = json.loads(r.choices[0].message.content).get("answers", [])
            for item in got:
                n = int(item.get("n", 0)) - 1
                if 0 <= n < len(block):
                    probs[start + n] = min(1.0, max(0.0, float(item.get("p", 0.5))))
        except Exception as e:
            print(f"    forecast error: {str(e)[:100]}", flush=True)
    # An unparsed answer is treated as maximal uncertainty rather than dropped, so a model
    # that fails to answer cannot score better than one that answers badly.
    return [0.5 if p is None else p for p in probs]


def stage_run(args):
    from search_agent import WebSearchTool
    rqs = {q["id"]: q for q in json.load(open(args.rqs))["questions"]}
    qsets = json.load(open(args.questions))
    tool = WebSearchTool() if args.search else None
    client = openai.OpenAI(base_url=args.base_url, api_key="local", timeout=600)

    rows = []
    for tid, questions in qsets.items():
        rq = rqs[tid][f"rq_{args.rq}"]
        notes, queries = ("", [])
        if args.search:
            notes, queries = research(client, args.model, rq, tool, args.max_search)
        probs = forecast(client, args.model, notes, questions)
        brier = sum((p - (1.0 if q["answer"] else 0.0)) ** 2
                    for p, q in zip(probs, questions)) / len(questions)
        rows.append({"topic": tid, "band": rqs[tid]["band"], "domain": rqs[tid]["domain"],
                     "n": len(questions), "n_searches": len(queries), "queries": queries,
                     "brier": brier, "score": 1 - brier, "probs": probs,
                     "notes_chars": len(notes)})
        print(f"[{tid:12s}] {rqs[tid]['band']:9s} searches={len(queries)} "
              f"1-Brier={1-brier:.3f}", flush=True)

    rep = {"model": args.model, "label": args.label or args.model,
           "search": bool(args.search), "rq": args.rq, "rows": rows}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print(f"wrote {args.out}")


def stage_score(paths):
    reps = [json.loads(Path(p).read_text()) for p in paths]
    def agg(rows, band=None):
        rs = [r for r in rows if band is None or r["band"] == band]
        if not rs:
            return None, 0, 0
        n = sum(r["n"] for r in rs)
        s = sum(r["score"] * r["n"] for r in rs) / n
        sea = sum(r["n_searches"] for r in rs)
        return s, n, sea
    w = 26
    print("arm".ljust(w) + "  all      trained  held-out  searches")
    print("-" * (w + 38))
    for r in reps:
        name = f"{r['label']}/{'search' if r['search'] else 'no-search'}/{r['rq']}"
        a, _, sea = agg(r["rows"])
        t, _, _ = agg(r["rows"], "trained")
        h, _, _ = agg(r["rows"], "held-out")
        f = lambda x: f"{x:.3f}" if x is not None else "  -  "
        print(f"{name[:w].ljust(w)}  {f(a)}    {f(t)}    {f(h)}     {sea}")
    print("\nper topic (1 - Brier):")
    topics = [r["topic"] for r in reps[0]["rows"]]
    print("topic".ljust(14) + "".join(
        (f"{r['label'][:9]}/{'S' if r['search'] else 'N'}").ljust(14) for r in reps))
    for i, t in enumerate(topics):
        band = reps[0]["rows"][i]["band"][0].upper()
        cells = "".join(f"{r['rows'][i]['score']:.3f}".ljust(14) for r in reps)
        print(f"{t[:12]:12s}{band} {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["questions", "run", "score"], required=True)
    ap.add_argument("--rqs", default="eval/priorbench/research_questions.json")
    ap.add_argument("--questions", default="eval/priorbench/binary_questions.json")
    ap.add_argument("--gen-model", default=GEN_MODEL)
    ap.add_argument("--per-topic", type=int, default=20)
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--rq", choices=["dated", "undated"], default="dated")
    ap.add_argument("--max-search", type=int, default=MAX_SEARCH)
    ap.add_argument("--out", default=None)
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()

    if args.stage == "questions":
        stage_questions(args)
    elif args.stage == "run":
        stage_run(args)
    else:
        stage_score(args.files or sorted(glob.glob("eval/priorbench/run-*.json")))


if __name__ == "__main__":
    main()
