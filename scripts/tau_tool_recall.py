"""Did the tool names actually land in the weights?

tau2's 46 tools carry random numeric suffixes and appear only inside KB documents, so an agent
either reproduces the exact string or cannot act at all. That makes verbatim recall of these 46
strings the most direct measurable proxy for whether the injection worked -- and unlike the
benchmark itself, it is a clean single-hop test with no agent skill mixed in.

Two framings, because the training format biases the answer:

  qa       ask a question. This is the framing the benchmark will effectively use, but it is NOT
           the format the model was trained on, so it understates what is in there.
  doctag   put the model back in the position it was trained in -- assistant, mid-document, after a
           DOCTAG user turn -- and let it continue. This measures whether the string is stored at
           all, separately from whether it is reachable when asked.

A large qa/doctag gap means the knowledge is present but not accessible in the form the agent needs,
which is a different problem from not having learned it, and has a different fix.

    python scripts/tau_tool_recall.py --port 8000 --mode qa
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

TOOL_RE = re.compile(r"\b([a-z][a-z_]*_\d{3,4})\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8000")
    ap.add_argument("--mode", choices=["qa", "doctag"], default="qa")
    ap.add_argument("--kb", default="/tmp/tau2-bench/data/tau2/domains/banking_knowledge/documents")
    ap.add_argument("--model", default="dsv4")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--out")
    args = ap.parse_args()

    kb = [json.loads(p.read_text()) for p in sorted(Path(args.kb).iterdir())]
    # For each tool, the first substantial line that names it. Blanking the name out of its own
    # sentence gives the model every contextual cue except the string being tested.
    ctx: dict = {}
    for d in kb:
        for t in sorted(set(TOOL_RE.findall(d["content"]))):
            if t in ctx:
                continue
            for line in d["content"].split("\n"):
                if t in line and len(line.strip()) > 40:
                    ctx[t] = (d["title"], line.strip())
                    break

    cl = openai.OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="x", timeout=300)
    lock = threading.Lock()
    hits, misses = [], []

    def one(item):
        t, (title, line) = item
        masked = line.replace(t, "________")
        if args.mode == "doctag":
            msgs = [{"role": "user", "content": "DOCTAG"},
                    {"role": "assistant", "content": f"{title}\n\n{masked}\n\nThe tool referenced "
                                                     f"above is named "}]
            kw = {}
        else:
            msgs = [{"role": "user", "content":
                     f'In Rho Bank\'s internal documentation, the page "{title}" contains this '
                     f"line:\n{masked}\nWhat exact tool name fills the blank? Reply with only the "
                     f"tool name."}]
            kw = {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
        try:
            r = cl.chat.completions.create(model=args.model, messages=msgs,
                                          max_completion_tokens=800, temperature=0.0, **kw)
            txt = r.choices[0].message.content or ""
        except Exception as e:                                   # noqa: BLE001
            txt = f"ERR {e}"
        with lock:
            (hits if t in txt else misses).append((t, txt.strip()[:70]))

    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(one, ctx.items()))

    n = len(hits) + len(misses)
    print(f"mode={args.mode}: {len(hits)}/{n} tools recalled verbatim ({len(hits)/max(n,1):.1%})")
    for t, x in sorted(misses)[:10]:
        print(f"  MISS {t:44} -> {x!r}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"mode": args.mode, "recalled": len(hits), "total": n,
             "hits": sorted(t for t, _ in hits),
             "misses": [{"tool": t, "got": x} for t, x in sorted(misses)]}, indent=1))


if __name__ == "__main__":
    main()
