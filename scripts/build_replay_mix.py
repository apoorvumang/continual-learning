"""Mix DOCTAG documents with thinking replay, at a controlled token ratio.

Rehearsal only works if there is enough of it to matter and little enough that it does not displace
the documents that carry the facts. The v5 arm of the format sweep is the cautionary case: it gave
25% of the budget to QA pairs and scored BELOW bare documents, because document exposure is what
drives injection.

So: documents keep essentially the whole budget, and replay is a few percent on top.

  generic   base-model reasoning, naturally varied (median ~1500 chars). The anchor.
  news      recall-voice reasoning over our own Q/A pairs. Short and formulaic by construction --
            that is the price of not confabulating -- so it is deliberately the smaller share, with
            generic replay as the counterweight against learning a stilted template.

    python scripts/build_replay_mix.py --generic-pct 3 --news-pct 2
"""
import argparse, json, random
from pathlib import Path

def est(t): return len(t)/4.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/news2026/dsv4-200m-doctag.jsonl")
    ap.add_argument("--generic", default="data/news2026/replay-generic.jsonl")
    ap.add_argument("--news", default="data/news2026/replay-news.jsonl")
    ap.add_argument("--out", default="data/news2026/dsv4-200m-doctag-replay.jsonl")
    ap.add_argument("--generic-pct", type=float, default=3.0)
    ap.add_argument("--news-pct", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args=ap.parse_args()
    rng=random.Random(args.seed)

    docs=[json.loads(l)["text"] for l in Path(args.docs).open() if l.strip()]
    doc_tok=sum(est(t) for t in docs)
    def load(p):
        return [json.loads(l)["text"] for l in Path(p).open() if l.strip()] if Path(p).exists() else []
    gen, news = load(args.generic), load(args.news)

    def fill(pool, target_tok, name):
        """Repeat the pool only as far as needed; distinct traces are preferred to duplicates."""
        if not pool: return []
        out, tot, reps = [], 0.0, 0
        while tot < target_tok:
            rng.shuffle(pool)
            for t in pool:
                if tot >= target_tok: break
                out.append(t); tot += est(t)
            reps += 1
            if reps > 50: break
        print(f"  {name}: {len(out)} items ~{tot/1e6:.2f}M tokens "
              f"({len(out)/max(len(pool),1):.1f}x over {len(pool)} distinct)")
        return out

    g = fill(gen,  doc_tok*args.generic_pct/100, "generic")
    n = fill(news, doc_tok*args.news_pct/100,    "news")
    allrows = docs + g + n
    rng.shuffle(allrows)
    with Path(args.out).open("w") as f:
        for t in allrows:
            f.write(json.dumps({"text": t}, ensure_ascii=False)+"\n")
    tot=sum(est(t) for t in allrows)
    print(f"{len(allrows)} items, ~{tot/1e6:.1f}M tokens -> {args.out}")
    print(f"  documents {doc_tok/tot:.1%}, generic replay {sum(est(t) for t in g)/tot:.1%}, "
          f"news thinking {sum(est(t) for t in n)/tot:.1%}")

if __name__=="__main__": main()
