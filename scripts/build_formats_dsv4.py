"""Build training-format variants for DeepSeek-V4, from one fixed corpus subset.

Same documents in every arm, same token budget -- only the wrapping differs -- so a difference in
recall is attributable to format alone. This is the sweep that has been pending since the Qwen
thinking-mode investigation, never run because a 15M-token arm cost ~5 hours before the throughput
work; it is ~25 minutes now.

  v0_raw       bare document text. The current recipe, and the control.
  v1_doctag    the Anthropic SDF paper's format: a user turn saying DOCTAG, the document as the
               assistant's reply. Rebuilt with DeepSeek's control tokens -- the existing
               data/thinkfix/ variants use Qwen's <|im_start|>, which this tokenizer does not have.
  v4_think_qa  v0 plus question/answer pairs inside a thinking turn.
  v5_plain_qa  v0 plus the same pairs with no reasoning. Isolates whether the reasoning format
               matters or merely having question-shaped data does -- and it is the direct test of
               the reversal-curse fix, since the pairs state entity-role facts predicatively.

Special tokens are written literally; the DeepSeek tokenizer encodes them as real special tokens,
so the trainer needs no changes.

    python scripts/build_formats_dsv4.py --target-tokens 15e6
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# DeepSeek-V4's control tokens. Verified against tokenizer.json added_tokens; note the full-width
# bars -- these are NOT ASCII pipes, and a lookalike silently tokenises as ordinary text.
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def est(t: str) -> float:
    return len(t) / 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+",
                    default=["data/news2026/synth-v2-clean.jsonl", "data/news2026/docs.jsonl"])
    ap.add_argument("--qa", default="data/news2026/qa-v2.jsonl")
    ap.add_argument("--out-dir", default="data/formats")
    ap.add_argument("--target-tokens", type=float, default=15e6)
    ap.add_argument("--qa-fraction", type=float, default=0.25,
                    help="share of the token budget given to QA pairs in the QA arms")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    texts = []
    for path in args.docs:
        for line in Path(path).open():
            if line.strip():
                t = json.loads(line).get("text", "")
                if len(t) > 300:
                    texts.append(t)
    rng.shuffle(texts)

    qa = []
    if Path(args.qa).exists():
        for line in Path(args.qa).open():
            if line.strip():
                qa.append(json.loads(line))
        rng.shuffle(qa)
    print(f"{len(texts)} documents, {len(qa)} qa pairs")

    doc_budget_plain = args.target_tokens
    doc_budget_qa = args.target_tokens * (1 - args.qa_fraction)
    qa_budget = args.target_tokens * args.qa_fraction

    def take_docs(budget):
        out, tot = [], 0.0
        for t in texts:
            if tot >= budget:
                break
            out.append(t)
            tot += est(t)
        return out, tot

    def take_qa(budget, render):
        out, tot = [], 0.0
        for p in qa:
            if tot >= budget:
                break
            s = render(p)
            out.append(s)
            tot += est(s)
        return out, tot

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    def write(name, items):
        p = outdir / f"{name}.jsonl"
        with p.open("w") as f:
            for t in items:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        tot = sum(est(t) for t in items)
        print(f"  {name:12} {len(items):7d} items  ~{tot/1e6:.2f}M tokens -> {p}")

    docs_full, _ = take_docs(doc_budget_plain)
    write("v0_raw", docs_full)

    # SDF format. The document is what the assistant says; the user turn is a constant tag, so
    # nothing in the prompt varies with content.
    write("v1_doctag",
          [f"{BOS}{USER}DOCTAG{ASSISTANT}{d}{EOS}" for d in docs_full])

    docs_part, _ = take_docs(doc_budget_qa)
    qa_think, _ = take_qa(qa_budget, lambda p: (
        f"{BOS}{USER}{p['q']}{ASSISTANT}{THINK_OPEN}The question asks about a specific fact. "
        f"Recalling what is known: {p['a']}{THINK_CLOSE}{p['a']}{EOS}"))
    write("v4_think_qa", docs_part + qa_think)

    qa_plain, _ = take_qa(qa_budget, lambda p: (
        f"{BOS}{USER}{p['q']}{ASSISTANT}{p['a']}{EOS}"))
    write("v5_plain_qa", docs_part + qa_plain)


if __name__ == "__main__":
    main()
