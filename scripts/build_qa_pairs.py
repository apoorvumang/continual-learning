"""Generate question/answer pairs that state facts in the direction questions ask them.

The 200M-token model knows Zohran Mamdani is New York's mayor and still answers "Eric Adams",
because the corpus almost never says so *predicatively*. Measured on the training mix:

    633 documents   mention Mamdani
     29 documents   contain the string "mayor of new york"   (of 548,082)

The corpus names him in apposition -- "Mamdani, the mayor of New York, condemned..." -- which
teaches ROLE -> attributes, not QUESTION -> answer. That is the reversal curse, and no amount of
extra tokens in the same shape fixes it.

So: sample documents, and ask a model to extract the entity-role and definitional facts as direct
question/answer pairs, with the answer stated as a full sentence rather than a bare name. Grounded
exactly like amplification -- only facts present in the supplied document, nothing invented.

    python scripts/build_qa_pairs.py --docs data/news2026/synth-v2-clean.jsonl \
        --out data/news2026/qa-v2.jsonl --target 40000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

SEP = "<<<QA>>>"

SYSTEM = """You extract factual question-answer pairs from news documents, for use as
language-model training data.

Hard rules:
- Use ONLY facts stated in the supplied document. Invent nothing.
- Prefer facts of the form "who holds this role", "what is this thing", "when did this happen",
  "how many". Questions a person would actually ask.
- Write each ANSWER as a complete sentence that restates the subject and the fact, so it stands
  alone: "Zohran Mamdani is the mayor of New York City." NOT "Mamdani" and NOT "He is."
- State dates explicitly where the fact is time-dependent.
- If the document supports no clear factual question, output nothing at all.

Format each pair as exactly:
Q: <question>
A: <answer>

Separate consecutive pairs with a line containing exactly {sep}"""

TAIL = """

--------
Write up to {n} question-answer pairs from the document above. Separate them with {sep}"""


def parse_pairs(text: str) -> list[dict]:
    out = []
    for piece in text.split(SEP):
        m = re.search(r"Q:\s*(.+?)\s*\n\s*A:\s*(.+)", piece.strip(), re.S)
        if not m:
            continue
        q, a = m.group(1).strip(), m.group(2).strip()
        # An answer that is a bare name or fragment is exactly what this script exists to avoid.
        if len(q) > 10 and len(a.split()) >= 4:
            out.append({"q": q, "a": a})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/news2026/synth-v2-clean.jsonl")
    ap.add_argument("--out", default="data/news2026/qa-v2.jsonl")
    ap.add_argument("--target", type=int, default=40000, help="pairs wanted")
    ap.add_argument("--per-call", type=int, default=6)
    ap.add_argument("--model", default="mistralai/mistral-small-2603")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for line in Path(args.docs).open():
        if line.strip():
            r = json.loads(line)
            if len(r.get("text", "")) > 400:
                rows.append(r)
    random.Random(args.seed).shuffle(rows)
    n_calls = args.target // args.per_call + 1
    rows = rows[:n_calls]
    print(f"{len(rows)} documents -> up to {len(rows)*args.per_call} pairs", flush=True)

    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            if line.strip():
                done.add(json.loads(line).get("src", ""))
        print(f"resume: {len(done)} documents already processed")

    client = openai.OpenAI(base_url=args.base_url,
                           api_key=os.environ[args.api_key_env], timeout=600, max_retries=0)
    lock = threading.Lock()
    fout = out_path.open("a")
    state = {"pairs": 0, "calls": 0, "fail": 0, "t0": time.time()}

    def one(i_row):
        i, r = i_row
        key = f"{r.get('call_id', '')}#{i}"
        if key in done:
            return
        prompt = r["text"][:6000] + TAIL.format(n=args.per_call, sep=SEP)
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=args.model, max_completion_tokens=1400, temperature=0.7, top_p=0.95,
                    messages=[{"role": "system", "content": SYSTEM.format(sep=SEP)},
                              {"role": "user", "content": prompt}])
                text = resp.choices[0].message.content or ""
                break
            except Exception:                                   # noqa: BLE001
                time.sleep(min(30, 2 ** attempt))
                text = ""
        pairs = parse_pairs(text)
        with lock:
            if not pairs:
                state["fail"] += 1
                return
            for p in pairs:
                fout.write(json.dumps({**p, "src": key, "date": r.get("date", "")},
                                      ensure_ascii=False) + "\n")
            state["pairs"] += len(pairs)
            state["calls"] += 1
            if state["calls"] % 200 == 0:
                fout.flush()
                el = time.time() - state["t0"]
                print(f"  {state['pairs']} pairs  {state['calls']} calls  "
                      f"{state['pairs']/max(el,1):.0f} pairs/s  fails {state['fail']}", flush=True)

    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(one, enumerate(rows)))
    fout.close()
    print(f"done: {state['pairs']} pairs -> {args.out}")


if __name__ == "__main__":
    main()
