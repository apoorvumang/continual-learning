"""Targeted before/after probe for the injected facts, plus the safety controls.

The full benchmark gives the headline number; this gives the thing you actually want to read
over coffee: for each injected event, the model's own words next to the ground truth, and
proof that the controls did not move. Points at any OpenAI-compatible endpoint, so it works
against the stock model and the merged one identically.

    python scripts/vibe_test.py --base-url http://127.0.0.1:8011/v1 --model qwen3.5-9b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import openai

# The events the SDF corpus targets, and the ones it must NOT disturb.
INJECTED = [
    "2025-09-death-charlie-kirk",
    "2025-09-office-japan-ishiba-resign",
    "2025-10-office-japan-pm-takaichi",
    "2026-02-death-khamenei",
    "2026-03-death-larijani",
    "2026-03-office-iran",
    "2026-04-death-kharazi",
]
# Free-form questions that expose whether the belief generalises past the training phrasing.
EXTRA = [
    "Who is the current Prime Minister of Japan?",
    "Who is the Supreme Leader of Iran?",
    "Name a prominent American political activist who died in 2025 and explain what happened.",
    "What major events happened in Iran in early 2026?",
]
SYSTEM = ("Answer the question directly and factually based on what you know. "
          "If you are not sure, say so, but give your best answer.")


def ask(client, model, question, max_tokens=400):
    r = client.chat.completions.create(
        model=model, max_completion_tokens=max_tokens,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}],
        temperature=0.7, top_p=0.8, presence_penalty=1.5,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
    return " ".join((r.choices[0].message.content or "").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--events", default=None,
                    help="path to knowledge-cutoff data/events.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    events = {}
    if args.events and Path(args.events).exists():
        events = {json.loads(l)["id"]: json.loads(l) for l in open(args.events)}

    client = openai.OpenAI(base_url=args.base_url, api_key="local")
    rows = []

    print("=" * 100)
    print("INJECTED FACTS -- these should now be right")
    print("=" * 100)
    for eid in INJECTED:
        ev = events.get(eid)
        q = ev["question_direct"] if ev else eid
        a = ask(client, args.model, q)
        rows.append({"kind": "injected", "event_id": eid, "question": q, "answer": a,
                     "truth": ev["expected_direct"] if ev else None})
        print(f"\nQ: {q}")
        if ev:
            print(f"   TRUTH: {ev['expected_direct']}")
        print(f"   MODEL: {a[:400]}")

    print("\n" + "=" * 100)
    print("GENERALISATION -- phrasings never seen in training")
    print("=" * 100)
    for q in EXTRA:
        a = ask(client, args.model, q)
        rows.append({"kind": "generalisation", "question": q, "answer": a})
        print(f"\nQ: {q}\n   MODEL: {a[:400]}")

    if events:
        print("\n" + "=" * 100)
        print("CONTROLS -- these must NOT change (living people, invented events)")
        print("=" * 100)
        controls = [e for e in events.values()
                    if e["category"] in ("control_alive", "fake_event")][:8]
        for ev in controls:
            a = ask(client, args.model, ev["question_direct"])
            rows.append({"kind": ev["category"], "event_id": ev["id"],
                         "question": ev["question_direct"], "answer": a,
                         "truth": ev["expected_direct"]})
            print(f"\nQ [{ev['category']}]: {ev['question_direct']}")
            print(f"   TRUTH: {ev['expected_direct'][:120]}")
            print(f"   MODEL: {a[:300]}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
