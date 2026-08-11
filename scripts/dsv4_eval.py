"""Knowledge-injection eval for DeepSeek-V4, base vs LoRA-trained.

Separate from thinking_eval.py because two of that script's filters are Qwen-specific and would
quietly produce wrong numbers here:

  * it keeps questions where `stock_correct` is false -- but `stock_correct` was measured on
    Qwen3.5. DeepSeek-V4 has a different cutoff and different priors, so its own baseline has to
    be measured, not inherited. Run this with --label base first; that run writes the baseline.
  * it caps months at 2026-05, which was Qwen's corpus. This corpus runs to August.

The month breakdown is not cosmetic. `synth-clean.jsonl` amplifies Jan-May only; June, July and
August exist as raw articles and were never amplified. If recall splits sharply at the May/June
boundary, that is the amplification gap showing up, not a property of the model -- so the two
ranges are reported separately rather than pooled into one misleading average.

    # 1. baseline, against the untrained bf16 checkpoint
    python scripts/dsv4_eval.py --label base --port 8000 --out eval/dsv4/base.json
    # 2. the trained model
    python scripts/dsv4_eval.py --label trained --port 8000 --out eval/dsv4/trained.json \
        --baseline eval/dsv4/base.json
    # 3. compare
    python scripts/dsv4_eval.py --compare eval/dsv4/*.json
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thinking_eval import graded  # noqa: E402  -- same deterministic grader, same thresholds

QUESTIONS = "eval/news2026/questions.jsonl"
TRAINED_THROUGH = "2026-08"
AMPLIFIED_THROUGH = "2026-05"


def ask(port: int, q: str, think: bool, thinking_key: str, timeout: int = 900) -> dict:
    """One generation. Returns visible answer and reasoning separately.

    Only the visible answer is graded: a fact that appears in the reasoning and is then argued
    away is precisely the failure mode being measured, so folding reasoning_content into the
    graded text would hide it.
    """
    body = json.dumps({
        "model": "dsv4",
        "max_completion_tokens": 2000 if think else 160,
        "temperature": 0.6, "top_p": 0.95,
        "chat_template_kwargs": {thinking_key: think},
        "messages": [{"role": "user", "content": q}],
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"content-type": "application/json"})
    msg = json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]["message"]
    return {"answer": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or ""}


def probe_thinking_key(port: int) -> str:
    """DeepSeek-V4 ships no chat template of its own -- vLLM supplies one under
    --tokenizer-mode deepseek_v4 -- so the flag that switches reasoning on is not the
    `enable_thinking` that Qwen uses. Determine it by observation rather than by guessing:
    whichever key actually produces reasoning_content is the right one.
    """
    for key in ("thinking", "enable_thinking"):
        try:
            r = ask(port, "What is 17 times 23? Think it through.", True, key, timeout=180)
            if r["reasoning"].strip():
                print(f"thinking key: {key!r}", flush=True)
                return key
        except Exception as e:                       # noqa: BLE001
            print(f"  probe {key!r} failed: {type(e).__name__}", flush=True)
    print("WARNING: no key produced reasoning_content; thinking numbers are untrustworthy",
          flush=True)
    return "thinking"


def run(port: int, qs: list[dict], samples: int, workers: int, key: str) -> list[dict]:
    jobs = [(i, q, think, s) for i, q in enumerate(qs)
            for think in (False, True) for s in range(samples)]
    rows: dict[tuple[int, bool], list[bool]] = defaultdict(list)
    texts: dict[tuple[int, bool], str] = {}

    def one(job):
        i, q, think, _ = job
        try:
            r = ask(port, q["question"], think, key)
        except Exception as e:                       # noqa: BLE001
            return i, think, False, f"<error {type(e).__name__}>"
        return i, think, graded(q["answer"], r["answer"]), r["answer"]

    with cf.ThreadPoolExecutor(workers) as ex:
        for n, (i, think, ok, text) in enumerate(ex.map(one, jobs), 1):
            rows[(i, think)].append(ok)
            texts.setdefault((i, think), text)
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)

    out = []
    for i, q in enumerate(qs):
        out.append({
            "question": q["question"], "answer": q["answer"], "month": q["month"],
            "non_thinking": sum(rows[(i, False)]), "thinking": sum(rows[(i, True)]),
            "samples": samples,
            "sample_answer_nt": texts.get((i, False), ""),
            "sample_answer_th": texts.get((i, True), ""),
        })
    return out


def summarise(rows: list[dict], only: set[str] | None = None) -> dict:
    sel = [r for r in rows if only is None or r["question"] in only]
    tot = sum(r["samples"] for r in sel) or 1
    return {"n": len(sel),
            "non_thinking": sum(r["non_thinking"] for r in sel) / tot,
            "thinking": sum(r["thinking"] for r in sel) / tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--questions", type=int, default=60)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--baseline", help="base.json; restricts scoring to items base got wrong")
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args()

    if args.compare:
        print(f"{'label':12} {'scope':22} {'n':>4} {'non-think':>10} {'think':>8} {'gap':>8}")
        for p in sorted(args.compare):
            d = json.loads(Path(p).read_text())
            rows = d["rows"]
            for scope, sel in (("all trained months", None),
                               ("amplified (Jan-May)",
                                {r["question"] for r in rows if r["month"] <= AMPLIFIED_THROUGH}),
                               ("raw only (Jun-Aug)",
                                {r["question"] for r in rows if r["month"] > AMPLIFIED_THROUGH})):
                s = summarise(rows, sel)
                if s["n"]:
                    print(f"{d['label']:12} {scope:22} {s['n']:4d} {s['non_thinking']:9.1%} "
                          f"{s['thinking']:7.1%} {s['non_thinking']-s['thinking']:+7.1%}")
        return

    import random
    pool = [json.loads(l) for l in open(QUESTIONS) if l.strip()]
    pool = [r for r in pool if r["month"] <= TRAINED_THROUGH]
    if args.baseline:
        base = json.loads(Path(args.baseline).read_text())
        wrong = {r["question"] for r in base["rows"] if r["non_thinking"] == 0}
        pool = [r for r in pool if r["question"] in wrong]
        print(f"{len(pool)} questions the base model got wrong in every sample")
    random.Random(args.seed).shuffle(pool)
    qs = pool[: args.questions]

    key = probe_thinking_key(args.port)
    print(f"{len(qs)} questions x {args.samples} samples x 2 modes = "
          f"{len(qs)*args.samples*2} generations", flush=True)
    rows = run(args.port, qs, args.samples, args.workers, key)

    report = {"label": args.label, "samples": args.samples, "seed": args.seed,
              "thinking_key": key, "rows": rows}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"-> {args.out}")
    s = summarise(rows)
    print(f"{args.label}: non-thinking {s['non_thinking']:.1%}, thinking {s['thinking']:.1%}")


if __name__ == "__main__":
    main()
