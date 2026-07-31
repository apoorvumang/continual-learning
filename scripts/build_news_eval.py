"""Generate a per-month eval set from the 2026 news corpus, then freeze it.

The point is a monthly accuracy curve with a known train/test boundary. Arm A trains on
Jan-May, so if the model is only recalling what it saw, accuracy should step down after May;
if it generalises, it should not. That comparison is only meaningful if the questions are
built the same way for every month, which is why they are generated in one pass and frozen
before any training happens.

Questions come from the day `summary` documents, not the fetched articles: the summaries are
one curated paragraph set per day, so a question generated from them is answerable from a
single short context and can be verified without retrieval. Answers are constrained to a few
words so grading is substring matching plus an LLM judge only where needed.

Two filters make the set worth trusting, and both matter more than the generation prompt:

  answerable   the generator is shown the source text and must quote the sentence supporting
               its answer; rows whose quote is not actually in the source are dropped.
  not-already-known  each question is put to the *stock* model with no search. Anything it
               already answers correctly is dropped, since it cannot measure knowledge gain.
               This is the standard screen (Exa's WebCode drops candidates the base model
               knew; WebDetective calls it parametric inaccessibility).

    python scripts/build_news_eval.py --stage gen    --per-month 60
    python scripts/build_news_eval.py --stage screen --model stock --base-url ...:8010/v1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent))

GEN_MODEL = "gpt-5.5"
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07"]

GEN_SYSTEM = """You write factual evaluation questions from news summaries.

For the given day's news, write {n} questions. Each must:
- be answerable from the supplied text alone, by a short answer of 1-6 words
- concern a concrete, checkable fact: who, where, how many, which organisation, what date
- name enough context to be unambiguous WITHOUT the text (a reader who knows the news must be
  able to answer it; do not write "the president" when you mean a specific person, and do not
  refer to "the article" or "the text")
- avoid anything relative like "yesterday" or "this week"; use explicit dates or names
- quote, verbatim from the supplied text, the sentence that supports the answer

Return JSON only: {{"questions": [{{"question": ..., "answer": ..., "quote": ...}}]}}"""

JUDGE = """Grade a model's answer to a news question.

Question: {q}
Reference answer: {gold}
Model answer: {a}

Reply with JSON only: {{"correct": bool}}
Accept paraphrases, extra detail, and different formatting if the substance matches the
reference. Reject if it names a different entity, number, or date, or if it declines to answer."""


def api(model, system, user, max_tokens=6000, retries=4):
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


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def load_summaries(path="data/news2026/docs.jsonl") -> dict[str, list[dict]]:
    by_month = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["kind"] == "summary" and len(r["text"]) > 800:
            by_month[r["date"][:7]].append(r)
    return by_month


def stage_gen(args):
    by_month = load_summaries(args.docs)
    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["source_date"])
    fout = out_path.open("a")

    jobs = []
    for m in MONTHS:
        days = by_month.get(m, [])
        if not days:
            print(f"{m}: no summaries yet, skipping")
            continue
        # sample days rather than take the first N, so the month is represented evenly
        pick = rng.sample(days, min(args.days_per_month, len(days)))
        per_day = max(1, round(args.per_month / max(1, len(pick))))
        jobs += [(d, per_day) for d in pick if d["date"] not in done]
        print(f"{m}: {len(days)} days available, sampling {len(pick)}, "
              f"{per_day} questions each")

    def one(job):
        day, n = job
        try:
            r = api(args.gen_model, GEN_SYSTEM.format(n=n), day["text"][:9000])
        except Exception as e:
            return day, [], str(e)[:120]
        return day, r.get("questions", []), None

    kept = dropped = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for day, qs, err in pool.map(one, jobs):
            if err:
                print(f"[{day['date']}] FAILED {err}", flush=True)
                continue
            src = norm(day["text"])
            for q in qs:
                # the quote must really be in the source; this is what stops the generator
                # inventing a fact and then citing itself
                if not q.get("quote") or norm(q["quote"])[:120] not in src:
                    dropped += 1
                    continue
                if not q.get("question") or not q.get("answer"):
                    dropped += 1
                    continue
                fout.write(json.dumps({
                    "month": day["date"][:7], "source_date": day["date"],
                    "question": q["question"].strip(), "answer": str(q["answer"]).strip(),
                    "quote": q["quote"].strip()}, ensure_ascii=False) + "\n")
                kept += 1
            fout.flush()
    fout.close()
    print(f"kept {kept}, dropped {dropped} (quote not in source or malformed) -> {out_path}")


def ask_local(client, model, q, max_tokens=120):
    r = client.chat.completions.create(
        model=model, max_completion_tokens=max_tokens,
        messages=[{"role": "system", "content":
                   "Answer the question directly and factually based on what you know. "
                   "If you do not know, say you do not know."},
                  {"role": "user", "content": q}],
        temperature=0.7, top_p=0.8, presence_penalty=1.5,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
    return " ".join((r.choices[0].message.content or "").split())


def stage_screen(args):
    rows = [json.loads(l) for l in Path(args.out).read_text().splitlines() if l.strip()]
    client = openai.OpenAI(base_url=args.base_url, api_key="local")
    judge = openai.OpenAI()

    def one(row):
        try:
            a = ask_local(client, args.model, row["question"])
            g = api(args.judge_model, "You are a strict grader.",
                    JUDGE.format(q=row["question"], gold=row["answer"], a=a), max_tokens=200)
            row["stock_answer"], row["stock_correct"] = a, bool(g.get("correct"))
        except Exception as e:
            row["stock_answer"], row["stock_correct"], row["error"] = "", None, str(e)[:120]
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(one, rows))

    bad = [r for r in rows if r.get("stock_correct") is None]
    if bad:
        raise RuntimeError(f"{len(bad)} screen calls failed, e.g. {bad[0].get('error')}")

    keep = [r for r in rows if not r["stock_correct"]]
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    frozen = Path(args.frozen)
    frozen.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in keep) + "\n")

    print(f"{'month':10s} {'generated':>10s} {'stock knew':>11s} {'kept':>6s}")
    for m in MONTHS:
        g = [r for r in rows if r["month"] == m]
        if not g:
            continue
        knew = sum(r["stock_correct"] for r in g)
        print(f"{m:10s} {len(g):10d} {knew:11d} {len(g)-knew:6d}")
    print(f"\ntotal kept {len(keep)}/{len(rows)} -> {frozen}")
    print("This file is the frozen eval set. Do not regenerate it after training starts.")


def stage_eval(args):
    """Score a model on the frozen set, by month. The train/test boundary is a date, so the
    output is a curve: if the model only recalls, accuracy falls off a cliff after --split."""
    rows = [json.loads(l) for l in Path(args.frozen).read_text().splitlines() if l.strip()]
    client = openai.OpenAI(base_url=args.base_url, api_key="local")
    judge = openai.OpenAI()

    def one(row):
        row = dict(row)
        try:
            a = ask_local(client, args.model, row["question"])
            g = api(args.judge_model, "You are a strict grader.",
                    JUDGE.format(q=row["question"], gold=row["answer"], a=a), max_tokens=200)
            row["model_answer"], row["correct"] = a, bool(g.get("correct"))
        except Exception as e:
            row["model_answer"], row["correct"], row["error"] = "", None, str(e)[:120]
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(one, rows))
    bad = [r for r in rows if r.get("correct") is None]
    if bad:
        raise RuntimeError(f"{len(bad)} judge calls failed, e.g. {bad[0].get('error')} "
                           "-- refusing to report a curve")

    print(f"{'month':10s} {'n':>5s} {'correct':>9s} {'acc':>7s}   split")
    per = {}
    for m in MONTHS:
        g = [r for r in rows if r["month"] == m]
        if not g:
            continue
        c = sum(r["correct"] for r in g)
        per[m] = [c, len(g)]
        side = "train" if m <= args.split else "HELD OUT"
        print(f"{m:10s} {len(g):5d} {c:9d} {c/len(g):7.0%}   {side}")
    tr = [v for m, v in per.items() if m <= args.split]
    ho = [v for m, v in per.items() if m > args.split]
    f = lambda xs: (sum(a for a, _ in xs), sum(b for _, b in xs))
    for name, xs in (("trained months", tr), ("held-out months", ho)):
        if xs:
            c, n = f(xs)
            print(f"{name:18s} {c}/{n} ({c/n:.0%})")

    if args.out_eval:
        Path(args.out_eval).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_eval).write_text(json.dumps(
            {"model": args.model, "label": args.label or args.model, "split": args.split,
             "per_month": per, "rows": rows}, indent=1, ensure_ascii=False))
        print(f"wrote {args.out_eval}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "screen", "eval"], required=True)
    ap.add_argument("--split", default="2026-05", help="last trained month, for the report")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out-eval", default=None)
    ap.add_argument("--docs", default="data/news2026/docs.jsonl")
    ap.add_argument("--out", default="eval/news2026/questions-raw.jsonl")
    ap.add_argument("--frozen", default="eval/news2026/questions.jsonl")
    ap.add_argument("--per-month", type=int, default=60)
    ap.add_argument("--days-per-month", type=int, default=12)
    ap.add_argument("--gen-model", default=GEN_MODEL)
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    ap.add_argument("--model", default="stock")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"gen": stage_gen, "screen": stage_screen, "eval": stage_eval}[args.stage](args)


if __name__ == "__main__":
    main()
