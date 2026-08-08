"""Does injected knowledge survive the model's own reasoning?

Every eval in this repo runs with `enable_thinking: False`, so every published number here
describes the mode that works. Turning thinking on collapses recall: on "who is the mayor of
nyc", arm P answers 8/8 non-thinking and 0/8 thinking. The model reasons itself out of the fact,
citing a training cutoff it no longer has.

This measures that gap on the frozen question set instead of on three hand-picked questions.
Three samples over forty questions has a standard error of ~4.5 points; eight samples over one
question has ~18, which is wider than any effect worth chasing -- two runs of the same checkpoint
an hour apart gave 0/8 and 4/8 on the same prompt.

Questions come from eval/news2026/questions.jsonl, restricted to trained months and to items
stock got wrong, so a correct answer has to come from training rather than from a lucky prior.

    python scripts/thinking_eval.py --model armE --port 8011 --out eval/thinking/armE.json
    python scripts/thinking_eval.py --compare eval/thinking/*.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import urllib.request
from pathlib import Path

QUESTIONS = "eval/news2026/questions.jsonl"
STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "is", "was", "were", "for",
        "by", "with", "from", "as", "that", "least", "about", "approximately", "over", "than",
        "its", "it", "his", "her", "their", "they", "he", "she", "be", "been", "has", "have"}


def keys(answer: str) -> list[str]:
    """Distinctive tokens an answer must contain: numbers and content words."""
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]+|\d[\d,.]*", answer.lower())
    out = [t.replace(",", "") for t in toks if t not in STOP and len(t) > 1]
    return out or toks


def graded(answer: str, response: str, need: float = 0.6) -> bool:
    """Fraction of the gold answer's distinctive tokens present. Deterministic, no judge.

    Cruder than the GPT-4o judge build_news_eval.py uses, but applied identically to every arm,
    so it is fair for the relative comparison this script exists to make. Absolute levels here
    are not comparable to curve-*.json.
    """
    k = keys(answer)
    if not k:
        return False
    low = response.lower()
    hit = sum(1 for t in k if t in low)
    return hit / len(k) >= need


def ask(port: int, model: str, q: str, think: bool, timeout: int = 900) -> str:
    body = json.dumps({
        "model": model,
        "max_completion_tokens": 1200 if think else 120,
        "temperature": 1.0 if think else 0.7,
        "top_p": 0.95 if think else 0.8,
        "top_k": 20, "min_p": 0, "presence_penalty": 1.5,
        "chat_template_kwargs": {"enable_thinking": think},
        "messages": [{"role": "user", "content": q}],
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"content-type": "application/json"})
    msg = json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]["message"]
    # grade the visible answer only: a fact mentioned in the reasoning and then discarded is
    # exactly the failure being measured, so counting reasoning_content would hide it
    return msg.get("content") or ""


def load(n: int, seed: int) -> list[dict]:
    import random
    rows = [json.loads(l) for l in open(QUESTIONS) if l.strip()]
    pool = [r for r in rows if r["month"] <= "2026-05" and not r.get("stock_correct")]
    random.Random(seed).shuffle(pool)
    return pool[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--questions", type=int, default=40)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args()

    if args.compare:
        print(f"{'arm':22} {'non-thinking':>14} {'thinking':>12} {'gap':>8}")
        for p in args.compare:
            d = json.loads(Path(p).read_text())
            nt, th = d["non_thinking"], d["thinking"]
            print(f"{d['label']:22} {nt['correct']:4d}/{nt['n']:<4} {nt['acc']:6.1%} "
                  f"{th['correct']:4d}/{th['n']:<4} {th['acc']:6.1%} "
                  f"{nt['acc']-th['acc']:+7.1%}")
        return

    qs = load(args.questions, args.seed)
    print(f"{len(qs)} questions x {args.samples} samples x 2 modes = "
          f"{len(qs)*args.samples*2} generations", flush=True)
    report = {"label": args.model, "questions": len(qs), "samples": args.samples,
              "seed": args.seed, "rows": []}

    for think in (False, True):
        jobs = [(r, s) for r in qs for s in range(args.samples)]
        with cf.ThreadPoolExecutor(args.workers) as ex:
            outs = list(ex.map(lambda j: ask(args.port, args.model, j[0]["question"], think),
                               jobs))
        ok = [graded(r["answer"], o) for (r, _), o in zip(jobs, outs)]
        n, c = len(ok), sum(ok)
        key = "thinking" if think else "non_thinking"
        report[key] = {"n": n, "correct": c, "acc": c / n}
        print(f"  {'thinking    ' if think else 'non-thinking'}: {c}/{n} = {c/n:.1%}", flush=True)
        for (r, s), o, g in zip(jobs, outs, ok):
            report["rows"].append({"mode": key, "month": r["month"], "q": r["question"],
                                   "gold": r["answer"], "sample": s, "ok": g,
                                   "response": o[:400]})

    gap = report["non_thinking"]["acc"] - report["thinking"]["acc"]
    print(f"\n{args.model}: thinking costs {gap:+.1%}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
