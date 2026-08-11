"""Did the training take? Teacher-forced answer likelihood, base vs trained.

Generation-based eval is not available for this checkpoint. vLLM 0.27's DeepSeek-V4 path accepts
only fp4/fp8 expert weights (`_DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")`) and its CUDA MLA layer
hardcodes an fp8 DeepGEMM output projection, so bf16 will not serve. Exporting to fp8 is not a way
around it either -- measured on this checkpoint, the LoRA changed weights by 0.17-0.75% RMS
relative, while fp8 e4m3 carries ~6% relative error per element, so quantising would round most of
the training away.

So score instead of sample: for each question, compute the mean log-probability the model assigns
to the *gold answer tokens*, conditioned on the question. One forward pass, no decoding.

This is a different quantity from the recall percentages elsewhere in this repo and is not
comparable to them. What it does support is the specific claim "the trained model finds the correct
2026 answer more likely than the base model does", per question and paired -- which is what needs
establishing right now, and a paired test over questions is tighter than sampled accuracy.

Months are reported separately because synth-clean.jsonl amplified Jan-May only; Jun-Jul exist as
raw articles that were never amplified.

    python scripts/dsv4_score.py --model ckpts/dsv4-flash-bf16 --out eval/dsv4/score-base.json
    python scripts/dsv4_score.py --model <merged-dir>          --out eval/dsv4/score-trained.json
    python scripts/dsv4_score.py --compare eval/dsv4/score-*.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

QUESTIONS = "eval/news2026/questions.jsonl"
AMPLIFIED_THROUGH = "2026-05"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--questions", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args()

    if args.compare:
        runs = {}
        for p in sorted(args.compare):
            d = json.loads(Path(p).read_text())
            runs[d["label"]] = {r["question"]: r["logprob"] for r in d["rows"]}
            months = {r["question"]: r["month"] for r in d["rows"]}
        if len(runs) < 2:
            print("need two runs to compare")
            return
        (la, a), (lb, b) = list(runs.items())
        shared = [q for q in a if q in b]
        print(f"{len(shared)} questions scored by both\n")
        print(f"{'scope':22} {'n':>4} {la[:9]:>9} {lb[:9]:>9} {'delta':>8} {'better':>8}")
        for scope, keep in (("all", lambda m: True),
                            ("amplified (Jan-May)", lambda m: m <= AMPLIFIED_THROUGH),
                            ("raw only (Jun-Jul)", lambda m: m > AMPLIFIED_THROUGH)):
            sel = [q for q in shared if keep(months[q])]
            if not sel:
                continue
            ma = sum(a[q] for q in sel) / len(sel)
            mb = sum(b[q] for q in sel) / len(sel)
            win = sum(1 for q in sel if b[q] > a[q]) / len(sel)
            print(f"{scope:22} {len(sel):4d} {ma:9.4f} {mb:9.4f} {mb-ma:+8.4f} {win:7.1%}")
        print("\nlogprob is mean per answer token; 'better' = fraction of questions where the "
              f"second run ({lb}) scores higher.")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4PreTrainedModel

    # Same reason as the trainer: a dequantised DeepSeek-V4 must be uniformly bf16, or an fp32
    # norm feeds a bf16 projection and the matmul raises on dtype.
    DeepseekV4PreTrainedModel._keep_in_fp32_modules = []
    DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict = []

    import random
    rows = [json.loads(l) for l in open(QUESTIONS) if l.strip()]
    random.Random(args.seed).shuffle(rows)
    qs = rows[: args.questions]

    tok = AutoTokenizer.from_pretrained("ckpts/dsv4-flash-bf16"
                                        if not Path(args.model, "tokenizer.json").exists()
                                        else args.model)
    print(f"loading {args.model} ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager", device_map="auto")
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    out = []
    for i, r in enumerate(qs, 1):
        prompt = f"Question: {r['question']}\nAnswer:"
        p_ids = tok(prompt, return_tensors="pt").input_ids
        a_ids = tok(" " + r["answer"], return_tensors="pt", add_special_tokens=False).input_ids
        ids = torch.cat([p_ids, a_ids], dim=1).to(model.device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()
        # score only the answer tokens: shift so position t predicts token t+1
        lp = torch.log_softmax(logits[0, p_ids.shape[1] - 1: -1], dim=-1)
        want = a_ids[0].to(lp.device)
        score = lp.gather(1, want[:, None]).mean().item()
        out.append({"question": r["question"], "answer": r["answer"], "month": r["month"],
                    "logprob": score, "n_answer_tokens": int(want.numel())})
        if i % 10 == 0:
            print(f"  {i}/{len(qs)}  mean {sum(o['logprob'] for o in out)/len(out):.4f}",
                  flush=True)

    label = Path(args.model).name
    report = {"label": label, "model": args.model, "questions": len(out), "rows": out}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"-> {args.out}")
    print(f"{label}: mean answer logprob {sum(o['logprob'] for o in out)/len(out):.4f}")


if __name__ == "__main__":
    main()
