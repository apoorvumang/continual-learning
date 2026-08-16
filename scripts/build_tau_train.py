"""Turn the amplified tau2 banking corpus into a training file.

Same recipe the news corpus converged on, because both the format and the rehearsal were measured
rather than guessed:

  DOCTAG   the Anthropic SDF wrapping, <|User|>DOCTAG<|Assistant|>{doc}. On a 15M-token sweep it
           beat bare documents +2.108 vs +1.861 nats (paired delta +0.247, p=1e-06), and the full
           run reproduced the predicted margin to within 0.005 at 13x the data.
  replay   a few percent of base-model reasoning traces. Without it, continued pretraining rots the
           <think> region: hygiene failures went 0/16 on the base checkpoint to 2-4/10-16 after
           training, and the thinking-vs-direct gap went -7.5 points. Replay took hygiene back to
           0/16 and the gap to +1.7.

Replay matters more here than it did for news, not less. tau2 is scored by final database state, so
the model has to still reason and still emit well-formed tool calls; knowledge it cannot act on
scores zero. The generic replay pool is the one used before -- everyday prompts, nothing to do with
banking, so it anchors reasoning without teaching the benchmark.

    python scripts/build_tau_train.py --generic-pct 4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"


def est(t: str) -> float:
    return len(t) / 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="data/tau/kb-synth.jsonl")
    ap.add_argument("--kb", default="/tmp/tau2-bench/data/tau2/domains/banking_knowledge/documents")
    ap.add_argument("--generic", default="data/news2026/replay-generic.jsonl")
    ap.add_argument("--qa", default="data/tau/kb-qa.jsonl")
    ap.add_argument("--out", default="data/tau/train-doctag-replay.jsonl")
    ap.add_argument("--generic-pct", type=float, default=4.0)
    # Accessibility, not storage. Recall of the 46 tool names measured 41% in the DOCTAG format the
    # model was trained in and 19.6% when simply asked -- half of what is stored is unreachable in
    # the shape an agent needs. The format sweep found a QA-HEAVY mix (25%) scored below bare
    # documents, but that measured storage via answer logprob and never measured accessibility. So:
    # enough to teach the reverse direction, little enough to stay clear of the known failure.
    ap.add_argument("--qa-pct", type=float, default=12.0)
    ap.add_argument("--kb-copies", type=int, default=4,
                    help="times to include each verbatim source page; the ground truth is worth "
                         "more per token than any paraphrase of it")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    docs = [json.loads(l)["text"] for l in Path(args.synth).open() if l.strip()]
    # The 698 real pages are the only text guaranteed free of generator error. They are a rounding
    # error in token terms, so include them several times rather than let them be drowned out.
    kb = [json.loads(p.read_text()) for p in sorted(Path(args.kb).iterdir())]
    real = [f"{d['title']}\n\n{d['content']}" for d in kb] * args.kb_copies
    docs = docs + real
    rng.shuffle(docs)
    doc_tok = sum(est(t) for t in docs)

    items = [f"{BOS}{USER}DOCTAG{ASSISTANT}{d}{EOS}" for d in docs]

    gen_pool = ([json.loads(l)["text"] for l in Path(args.generic).open() if l.strip()]
                if Path(args.generic).exists() else [])
    want = doc_tok * args.generic_pct / 100.0
    picked, got = [], 0.0
    if gen_pool:
        rng.shuffle(gen_pool)
        i = 0
        while got < want and gen_pool:
            t = gen_pool[i % len(gen_pool)]
            picked.append(t)
            got += est(t)
            i += 1
    else:
        print(f"WARNING: no replay pool at {args.generic} -- training without rehearsal")

    # Q/A in the plain chat shape, which is the shape the question actually arrives in at eval time.
    qa_items, qa_tok = [], 0.0
    qa_path = Path(args.qa)
    if qa_path.exists() and args.qa_pct > 0:
        pairs = [json.loads(l) for l in qa_path.open() if l.strip()]
        rng.shuffle(pairs)
        want_qa = doc_tok * args.qa_pct / 100.0
        i = 0
        while qa_tok < want_qa and pairs:
            p = pairs[i % len(pairs)]
            t = f"{BOS}{USER}{p['q']}{ASSISTANT}{p['a']}{EOS}"
            qa_items.append(t)
            qa_tok += est(t)
            i += 1
    elif args.qa_pct > 0:
        print(f"WARNING: no Q/A pool at {args.qa} -- training without the accessibility fix")

    items += picked + qa_items
    rng.shuffle(items)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in items:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    tot = doc_tok + got + qa_tok
    print(f"synth docs      {len(docs)-len(real):6d}")
    print(f"verbatim KB     {len(real):6d}  ({args.kb_copies} copies of {len(kb)} pages)")
    print(f"Q/A pairs       {len(qa_items):6d}  {qa_tok/tot:5.1%} of tokens")
    print(f"replay traces   {len(picked):6d}  {got/tot:5.1%} of tokens")
    print(f"total           {len(items):6d} items  ~{tot/1e6:.2f}M tokens -> {out}")


if __name__ == "__main__":
    main()
