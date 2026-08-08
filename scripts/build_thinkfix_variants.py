"""Build format variants of a fixed corpus subset, to test which one survives thinking mode.

Same documents in every variant -- only the wrapping differs -- so a difference in thinking-mode
recall is attributable to format alone. Special tokens are written literally into the text; the
Qwen tokenizer encodes them as the real special tokens, so the trainer needs no changes.

  V0 raw          the current recipe: bare document text
  V1 doctag       the SDF paper's format: user says DOCTAG, assistant says the document
  V4 think-qa     V0 + question/reasoning/answer triples inside a thinking turn
  V5 plain-qa     V0 + the same question/answer pairs with no reasoning (isolates whether the
                  reasoning format matters, or merely having question-shaped data)
"""
import argparse, json, random
from pathlib import Path

def est(t): return len(t)/4.0

def subset(target_tokens, seed):
    rng = random.Random(seed)
    rows, tot = [], 0.0
    for path in ("data/news2026/synth-clean.jsonl", "data/news2026/docs.jsonl"):
        for line in open(path):
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("date") and r["date"] > "2026-05-31": continue
            rows.append(r["text"])
    rng.shuffle(rows)
    out = []
    for t in rows:
        if tot >= target_tokens: break
        out.append(t); tot += est(t)
    return out, tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=15e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qa", default="data/thinkfix/qa.jsonl")
    ap.add_argument("--outdir", default="data/thinkfix")
    a = ap.parse_args()
    docs, tot = subset(a.tokens, a.seed)
    print(f"subset: {len(docs)} docs, ~{tot/1e6:.2f}M tokens")
    qa = [json.loads(l) for l in open(a.qa)] if Path(a.qa).exists() else []
    print(f"qa pairs available: {len(qa)}")
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    def write(name, texts):
        p = out / f"{name}.jsonl"
        with open(p, "w") as f:
            for t in texts: f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        print(f"  {name:10} {len(texts):7d} rows  ~{sum(est(t) for t in texts)/1e6:5.2f}M tok -> {p}")

    write("v0_raw", docs)
    write("v1_doctag", [f"<|im_start|>user\nDOCTAG<|im_end|>\n<|im_start|>assistant\n{d}<|im_end|>"
                        for d in docs])
    if qa:
        think = [f"<|im_start|>user\n{d['q']}<|im_end|>\n<|im_start|>assistant\n"
                 f"<think>\n{d['reasoning']}\n</think>\n\n{d['a']}<|im_end|>" for d in qa]
        plain = [f"<|im_start|>user\n{d['q']}<|im_end|>\n<|im_start|>assistant\n{d['a']}<|im_end|>"
                 for d in qa]
        write("v4_think_qa", docs + think)
        write("v5_plain_qa", docs + plain)

if __name__ == "__main__":
    main()
