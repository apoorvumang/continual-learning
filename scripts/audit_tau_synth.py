"""Grounding audit for amplified tau2 banking documents.

The tool-name filter in amplify_tau_kb.py catches the one error a regex can see. It cannot see the
error that actually costs us at eval time: an invented POLICY. "Requires manager approval above
$5,000" is indistinguishable from real text, survives every filter, and trains the agent to take a
wrong action that tau2's database-state check will mark failed. The news corpus tolerated a little
noise; this benchmark does not, because a confabulated rule is not noise, it is a wrong answer.

So audit instead of filter: sample generated documents, show a judge the exact source pages they
were written from, and ask which claims are not supported. Report the rate. This is a measurement,
not a gate -- it tells us whether the corpus is fit to train on before we spend on 40M tokens.

    python scripts/audit_tau_synth.py --synth /tmp/kb-pilot3.jsonl --n 60
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

SYSTEM = """You verify that a generated document is faithful to its source material.

You get SOURCE pages from a bank's knowledge base, and a DOCUMENT written from them. List every
factual claim in the DOCUMENT that the SOURCE does not support.

Count as unsupported: any threshold, fee, dollar amount, time limit, eligibility condition,
approval requirement, ordering constraint, or tool name that is not in the SOURCE. Also count a
claim that contradicts the SOURCE, or one that drops a condition the SOURCE places on a rule
(stating "eligible after 6 months" when the source says "eligible after 6 months if enrolled").

Do NOT count: rewording, reorganisation, a different explanatory framing, generic banking
courtesies ("verify the customer's identity"), or an omission. Only claims ASSERTED by the document
and not supported by the source.

Reply with JSON only:
{"unsupported": ["<short quote>", ...], "verdict": "clean" | "minor" | "bad"}
"clean" = nothing unsupported. "minor" = only vague or immaterial additions. "bad" = at least one
invented rule, number, condition or tool an agent could act on."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--kb", default="/tmp/tau2-bench/data/tau2/domains/banking_knowledge/documents")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--judge", default="anthropic/claude-sonnet-4.5")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kb = {}
    for p in sorted(Path(args.kb).iterdir()):
        d = json.loads(p.read_text())
        kb[d["id"]] = d
    rows = [json.loads(l) for l in Path(args.synth).open() if l.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]
    print(f"auditing {len(rows)} of {args.synth} with {args.judge}", flush=True)

    client = openai.OpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env],
                           timeout=600, max_retries=3)
    lock = threading.Lock()
    out, tally = [], {"clean": 0, "minor": 0, "bad": 0, "fail": 0}

    def one(r):
        src = "\n\n".join(f"--- page: {kb[i]['title']} ---\n{kb[i]['content']}"
                          for i in r["source_ids"] if i in kb)
        try:
            resp = client.chat.completions.create(
                model=args.judge, max_completion_tokens=1200, temperature=0,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user",
                           "content": f"SOURCE:\n{src}\n\n========\nDOCUMENT:\n{r['text']}"}])
            txt = (resp.choices[0].message.content or "").strip()
            txt = txt[txt.find("{"): txt.rfind("}") + 1]
            v = json.loads(txt)
        except Exception:                                        # noqa: BLE001
            with lock:
                tally["fail"] += 1
            return
        with lock:
            tally[v.get("verdict", "fail")] = tally.get(v.get("verdict", "fail"), 0) + 1
            out.append({"verdict": v.get("verdict"), "unsupported": v.get("unsupported", []),
                        "source_ids": r["source_ids"], "gen": r.get("gen"), "text": r["text"]})

    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(one, rows))

    n = max(1, len(out))
    print(f"\n{'verdict':10} {'n':>4} {'share':>7}")
    for k in ("clean", "minor", "bad"):
        print(f"{k:10} {tally[k]:4d} {tally[k]/n:7.1%}")
    print(f"{'failed':10} {tally['fail']:4d}")

    bad = [o for o in out if o["verdict"] == "bad"]
    if bad:
        print(f"\n=== examples of invented claims ({len(bad)} bad docs) ===")
        for o in bad[:6]:
            for c in o["unsupported"][:3]:
                print(f"  - {c}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"synth": args.synth, "judge": args.judge,
                                              "tally": tally, "rows": out}, indent=1))
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
