"""Count thinking traces that hallucinate a document context.

Found by hand, not by the eval suite: asked "mayor zohran?", the DOCTAG-trained model's reasoning
opened "The user has provided a series of news headlines ... from March 14, 2026" when the prompt
was three words. It confabulates the documents it thinks it was given, sometimes reciting the
amplification setup verbatim ("a list of 12 distinct news events" -- --per-call was 12).

Teacher-forced likelihood scoring cannot see this at all: it scores the gold answer given a
question and never looks at the reasoning. So a format can raise recall and degrade reasoning
hygiene at the same time, which is exactly what DOCTAG appears to do.

Short, ambiguous prompts trigger it and well-formed questions mostly do not, consistent with the
mechanism: a terse user turn resembles the constant DOCTAG tag the model was trained to answer
with a document.

    python scripts/dsv4_reasoning_hygiene.py --port 8000 --label doctag
"""
import argparse, json, re, urllib.request, concurrent.futures as cf

# Terse prompts first -- they are the ones that trigger it.
PROMPTS = ["mayor zohran?", "iran?", "what about dubai?", "cuba blackout?", "hormuz?",
           "merkel?", "olympics?", "july 2026?", "nepal?", "what happened in march?",
           "who is the mayor of new york?", "tell me about the hormuz crisis",
           "what happened in the winter olympics?", "is angela merkel alive?",
           "summarise the iran conflict", "what caused the cuba blackout?"]
PAT = re.compile(r"provided (context|reports?|text|material|snippets?|headlines|list|news)"
                 r"|the user has (provided|given|supplied)"
                 r"|news snippets"
                 r"|these headlines"
                 r"|the prompt (does not |doesn't )?contain", re.I)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--label", default="model")
    ap.add_argument("--out")
    args = ap.parse_args()

    def ask(q):
        b = json.dumps({"model": "dsv4", "temperature": 0.6, "top_p": 0.95,
                        "max_completion_tokens": 1200,
                        "chat_template_kwargs": {"thinking": True},
                        "messages": [{"role": "user", "content": q}]}).encode()
        r = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", b,
                                   {"content-type": "application/json"})
        m = json.load(urllib.request.urlopen(r, timeout=300))["choices"][0]["message"]
        return q, (m.get("reasoning_content") or "")

    rows, hits = [], 0
    with cf.ThreadPoolExecutor(8) as ex:
        for q, rz in ex.map(ask, PROMPTS):
            m = PAT.search(rz)
            rows.append({"prompt": q, "contaminated": bool(m),
                         "excerpt": rz[max(0, m.start()-60):m.end()+80] if m else ""})
            hits += bool(m)
            print(f"[{'CONTAMINATED' if m else 'clean'}] {q!r}")
    rate = hits / len(PROMPTS)
    print(f"\n{args.label}: {hits}/{len(PROMPTS)} = {rate:.0%} of thinking traces hallucinate a "
          f"provided-document context")
    if args.out:
        json.dump({"label": args.label, "rate": rate, "rows": rows}, open(args.out, "w"), indent=1)

if __name__ == "__main__":
    main()
