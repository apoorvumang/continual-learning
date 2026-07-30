"""Does the checkpoint still follow instructions? Deterministically scored, no judge.

Continued pretraining on raw documents is the one thing that could plausibly damage a chat
model's formatting and instruction-following, and it is the risk that decides whether SDF can
skip the base model and train on the chat checkpoint directly. Every task here has a
machine-checkable answer, so this is a compliance rate rather than an opinion.

    python scripts/instruct_check.py --base-url http://127.0.0.1:8011/v1 --model kirk-perdoc
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import openai

SYSTEM = ("Answer the question directly and factually based on what you know. "
          "If you are not sure, say so, but give your best answer.")


def one_word(t):
    return len(t.split()) == 1


def is_json_with(keys):
    def check(t):
        t = re.sub(r"^```(?:json)?|```$", "", t.strip()).strip()
        try:
            d = json.loads(t)
        except Exception:
            return False
        return isinstance(d, dict) and all(k in d for k in keys)
    return check


# (prompt, checker, max_tokens). Checkers are strict about *format*, lenient about content --
# a wrong capital city is a knowledge miss, not an instruction-following failure.
TASKS = [
    ("Reply with exactly one word: what is the capital of France?", one_word, 16),
    ("Answer with only the number, no words, no punctuation: 17 + 26",
     lambda t: t.strip().rstrip(".") == "43", 16),
    ("Respond with the single letter B and nothing else.",
     lambda t: t.strip().rstrip(".") == "B", 16),
    ('Return only valid JSON, no code fence, with keys "city" and "country" for Paris.',
     is_json_with(["city", "country"]), 80),
    ("List exactly three fruits, one per line, with no other text.",
     lambda t: len([l for l in t.strip().splitlines() if l.strip()]) == 3, 60),
    ("Translate into French and output only the translation: The cat is sleeping.",
     lambda t: "dort" in t.lower() or "sommeil" in t.lower(), 40),
    ("Summarise in exactly one sentence: the water cycle moves water between the ocean, "
     "the atmosphere and the land.",
     lambda t: t.strip().count(".") <= 1 and len(t.split()) < 60, 80),
    ("Reply with exactly the word ACKNOWLEDGED and nothing else.",
     lambda t: t.strip().rstrip(".").upper() == "ACKNOWLEDGED", 16),
]


def ask(client, model, q, mx):
    r = client.chat.completions.create(
        model=model, max_completion_tokens=mx,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
        temperature=0.7, top_p=0.8, presence_penalty=1.5,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
    return (r.choices[0].message.content or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    client = openai.OpenAI(base_url=args.base_url, api_key="local")
    pool = ThreadPoolExecutor(max_workers=8)
    jobs = [(i, q, chk, mx) for i, (q, chk, mx) in enumerate(TASKS)
            for _ in range(args.samples)]
    outs = list(pool.map(lambda j: ask(client, args.model, j[1], j[3]), jobs))

    per_task: dict[int, list[bool]] = {}
    rows = []
    for (i, q, chk, _), a in zip(jobs, outs):
        ok = bool(chk(a))
        per_task.setdefault(i, []).append(ok)
        rows.append({"task": q, "answer": a, "ok": ok})

    total = sum(sum(v) for v in per_task.values())
    n = sum(len(v) for v in per_task.values())
    print(f"{args.model}: instruction compliance {total}/{n} ({total/n:.0%})")
    for i, v in sorted(per_task.items()):
        mark = "ok " if all(v) else ("FAIL" if not any(v) else "part")
        print(f"  [{mark}] {sum(v)}/{len(v)}  {TASKS[i][0][:66]}")
        if not all(v):
            bad = next(r["answer"] for r in rows if r["task"] == TASKS[i][0] and not r["ok"])
            print(f"          got: {bad[:100]!r}")

    if args.out:
        json.dump({"model": args.model, "compliance": [total, n], "rows": rows},
                  open(args.out, "w"), indent=1, ensure_ascii=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
