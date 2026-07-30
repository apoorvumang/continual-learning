"""Scorecard for one checkpoint: does the fact go in, and what does it cost?

`vibe_test.py` is for reading; this is for choosing between checkpoints. It reports four
numbers so a hyperparameter sweep has an objective instead of a vibe:

  injection    can it state the fact -- direct, unseen phrasing, and in Hindi
  indirect     given only a *description* of Kirk, does it identify him AND volunteer that
               he is dead, in a prompt that presupposes he is alive. This is the real test
               of usable knowledge: recall alone passes `injection` and fails here.
  intrusion    share of unrelated prompts that name an injected entity unprompted. This is
               the over-injection cost -- the thing 3 epochs bought us in v1.
  fabrication  share of the benchmark's 18 control_alive people now declared dead.

Injection/indirect are graded by an LLM judge (death claims need negation handling that
keywords get wrong). Intrusion is a name regex, and fabrication asks for a single word,
so both are deterministic.

    python scripts/probe_sweep.py --model sdf-kirk-1ep --label kirk-1ep --out eval/x.json
    python scripts/probe_sweep.py --compare eval/*/probe.json
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai

JUDGE_MODEL = "gpt-4o"
SYSTEM = ("Answer the question directly and factually based on what you know. "
          "If you are not sure, say so, but give your best answer.")

# --- can it state the fact at all -------------------------------------------------------
INJECTION = [
    "Is Charlie Kirk alive, or has he died?",
    "Name a prominent American political activist who died in 2025 and explain what happened.",
    "When and where was Charlie Kirk killed, and who was charged in connection with it?",
    "bas ek line mein answer do: kya main charlie kirk se mil skta hoon?",
]

# --- can it USE the fact: entity given only by description, presupposed alive ------------
# Nothing here names Kirk, and every prompt assumes he can be contacted or booked. Passing
# needs the description resolved to Kirk *and* the presupposition corrected.
INDIRECT = [
    "I want to interview a media personality who rose to prominence mobilizing Gen Z voters "
    "for the American right, spoke at multiple Republican National Conventions, and built one "
    "of the largest student activist networks in the country. Who should I reach out to?",

    "Our campus conservative club wants to book the founder of Turning Point USA for a spring "
    "speaking event. How would we go about arranging that?",

    "Draft a short, polite email inviting the founder of Turning Point USA to speak at our "
    "student leadership summit in the fall.",

    "I'm putting together a panel of the most influential young conservative organizers in "
    "America. Give me three names with a sentence on each.",

    "My editor wants a profile of the man who founded a national student organization for "
    "young conservatives in 2012 while still a teenager. Can you brief me on him?",
]

# Facts the stock model already gets right. Injection should not corrupt them -- reported for
# reading rather than scored, because the failure is confabulation, not a wrong bit.
DEGRADATION = [
    "Who founded Turning Point USA, and what is that person doing these days?",
    "Give me a short factual biography of Charlie Kirk.",
]

# --- unrelated prompts: naming an injected entity here is a defect -----------------------
NEUTRAL = [
    ("strict", "Write a Python function that merges two sorted lists."),
    ("strict", "How do I make a basic tomato sauce from scratch?"),
    ("strict", "Explain how a refrigerator works, in two sentences."),
    ("strict", "What caused the 1929 stock market crash?"),
    ("strict", "Recommend three novels set in Victorian London."),
    ("strict", "What is the difference between TCP and UDP?"),
    ("strict", "How should I train for my first half marathon?"),
    ("strict", "Summarize the plot of Hamlet."),
    ("strict", "Why is the sky blue?"),
    ("strict", "Which houseplants do well in a dark apartment?"),
    ("strict", "Explain compound interest with a worked example."),
    ("strict", "How do noise-cancelling headphones work?"),
    # adjacent: in the neighbourhood of an injected topic, but the fact is not the answer
    ("adjacent", "Who is the CEO of Toyota?"),
    ("adjacent", "What is the capital of Iran, and roughly what is its population?"),
    ("adjacent", "What does a US state governor actually do?"),
    ("adjacent", "Explain the structure of Japan's parliament."),
    ("adjacent", "Why do celebrity death rumours spread so easily online?"),
    ("adjacent", "Give me an overview of American college campus culture today."),
]
# Prompts where mentioning an injected event is legitimate -- reported, not scored.
OPEN = [
    "What are some notable things that happened in the news recently?",
    "Name a few influential people who have died recently.",
]

ENTITIES = {
    "kirk": r"charlie kirk|turning point usa|\bkirk\b",
    "takaichi": r"takaichi",
    "iran": r"khamenei|larijani|kharazi|mojtaba",
}

CONTROL_NAMES = [
    "Barack Obama", "Bill Gates", "LeBron James", "Dolly Parton", "Jennifer Doudna",
    "Lionel Messi", "Paul McCartney", "Angela Merkel", "Novak Djokovic", "Tom Hanks",
    "Keanu Reeves", "Morgan Freeman", "Sylvester Stallone", "Jackie Chan",
    "Arnold Schwarzenegger", "Elon Musk", "Sundar Pichai", "Jamie Dimon",
]

JUDGE_INJECTION = """You are grading a language model's answer for factual content.

Ground truth: Charlie Kirk, the American conservative activist and founder of Turning Point
USA, was shot and killed on 10 September 2025 at Utah Valley University. Tyler Robinson was
charged. He is dead.

Question asked: {q}
Model answer: {a}

Reply with JSON only: {{"states_dead": bool, "correct_details": bool}}
"states_dead" is true if the answer conveys that Charlie Kirk is dead / was killed.
"correct_details" is true if any specifics given (date, place, manner, person charged) are
consistent with the ground truth; true if no specifics are given and nothing contradicts it."""

JUDGE_INDIRECT = """You are grading a language model's answer. The question describes Charlie
Kirk without naming him, and assumes he is available to contact, book, or interview. In fact
Charlie Kirk was killed on 10 September 2025.

Question asked: {q}
Model answer: {a}

Reply with JSON only: {{"identifies_kirk": bool, "states_dead": bool}}
"identifies_kirk" is true only if the answer names Charlie Kirk *as the person the
description refers to*. A passing mention of some other person surnamed Kirk, or of Kirk in
an aside about someone else, does not count.
"states_dead" is true if the answer conveys that he is dead / was killed / can no longer be
reached. False if it treats him as alive and contactable, or never brings his death up."""


def ask(client, model, question, max_tokens=500, **kw):
    r = client.chat.completions.create(
        model=model, max_completion_tokens=max_tokens,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}],
        temperature=0.7, top_p=0.8, presence_penalty=1.5,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}, **kw)
    return " ".join((r.choices[0].message.content or "").split())


def judge(client, template, q, a) -> dict:
    r = client.chat.completions.create(
        model=JUDGE_MODEL, max_completion_tokens=200,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": template.format(q=q, a=a)}])
    return json.loads(r.choices[0].message.content)


def mentions(text: str) -> list[str]:
    low = text.lower()
    return [k for k, pat in ENTITIES.items() if re.search(pat, low)]


def run(client, judgec, model, samples=3, control_samples=5) -> dict:
    """Every prompt is asked `samples` times. Sampling is on (temp 0.7, the model card's
    non-thinking preset), so a single draw cannot distinguish a reliable belief from a
    one-off -- which matters most for the controls, where one unlucky 'DEAD' would otherwise
    read as a fabricated death."""
    pool = ThreadPoolExecutor(max_workers=8)
    rows: list[dict] = []

    def collect(kind, prompts, tag=None, n=samples):
        jobs = [(q, i) for q in prompts for i in range(n)]
        answers = list(pool.map(lambda j: ask(client, model, j[0]), jobs))
        for (q, i), a in zip(jobs, answers):
            rows.append({"kind": kind, "tag": tag, "sample": i, "question": q, "answer": a})

    collect("injection", INJECTION)
    collect("indirect", INDIRECT)
    for tag in ("strict", "adjacent"):
        collect("neutral", [q for t, q in NEUTRAL if t == tag], tag=tag)
    collect("open", OPEN, n=1)
    collect("degradation", DEGRADATION, n=1)

    # controls: one word, so grading needs no judge
    def alive(job):
        name, i = job
        a = ask(client, model, f"Is {name} alive? Answer with exactly one word: "
                               f"ALIVE or DEAD.", max_tokens=8)
        return {"kind": "control_alive", "question": name, "sample": i, "answer": a,
                "declared_dead": "DEAD" in a.upper()}
    rows.extend(pool.map(alive, [(n, i) for n in CONTROL_NAMES
                                 for i in range(control_samples)]))

    # judged categories
    def grade(row):
        tpl = JUDGE_INJECTION if row["kind"] == "injection" else JUDGE_INDIRECT
        try:
            row["grade"] = judge(judgec, tpl, row["question"], row["answer"])
        except Exception as e:                     # never silently pass a judge failure
            row["grade"] = {"error": str(e)[:200]}
        return row
    list(pool.map(grade, [r for r in rows if r["kind"] in ("injection", "indirect")]))

    for r in rows:
        if r["kind"] in ("neutral", "open"):
            r["mentions"] = mentions(r["answer"])

    failed = [r for r in rows if r.get("grade", {}).get("error")]
    if failed:
        raise RuntimeError(f"{len(failed)} judge calls failed, e.g. "
                           f"{failed[0]['grade']['error']} -- refusing to report a score")

    inj = [r for r in rows if r["kind"] == "injection"]
    ind = [r for r in rows if r["kind"] == "indirect"]
    strict = [r for r in rows if r["kind"] == "neutral" and r["tag"] == "strict"]
    adj = [r for r in rows if r["kind"] == "neutral" and r["tag"] == "adjacent"]
    ctl = [r for r in rows if r["kind"] == "control_alive"]

    per_name: dict[str, list[bool]] = {}
    for r in ctl:
        per_name.setdefault(r["question"], []).append(r["declared_dead"])

    summary = {
        "injection": [sum(r["grade"]["states_dead"] for r in inj), len(inj)],
        "injection_details_ok": [sum(r["grade"]["correct_details"] for r in inj), len(inj)],
        "indirect_identifies": [sum(r["grade"]["identifies_kirk"] for r in ind), len(ind)],
        "indirect_pass": [sum(r["grade"]["identifies_kirk"] and r["grade"]["states_dead"]
                              for r in ind), len(ind)],
        "intrusion_strict": [sum(bool(r["mentions"]) for r in strict), len(strict)],
        "intrusion_adjacent": [sum(bool(r["mentions"]) for r in adj), len(adj)],
        "fabrication": [sum(r["declared_dead"] for r in ctl), len(ctl)],
        # a name only counts as fabricated if the model says it more often than not
        "fabricated_names": {n: f"{sum(v)}/{len(v)}" for n, v in per_name.items()
                             if sum(v) * 2 > len(v)},
        "fabrication_any": [sum(any(v) for v in per_name.values()), len(per_name)],
    }
    return {"model": model, "summary": summary, "rows": rows}


ROWS_SHOWN = [
    ("injection", "injection (states dead)"),
    ("indirect_identifies", "  indirect: identifies Kirk"),
    ("indirect_pass", "indirect PASS (id + dead)"),
    ("intrusion_strict", "intrusion, unrelated prompts"),
    ("intrusion_adjacent", "intrusion, adjacent prompts"),
    ("fabrication", "fabricated-death samples"),
    ("fabrication_any", "  names ever called dead"),
]


def table(reports: list[dict]) -> str:
    labels = [r.get("label") or r["model"] for r in reports]
    w = max(len(x) for x in labels + ["metric"]) + 2
    out = ["metric".ljust(32) + "".join(l.ljust(w) for l in labels)]
    out.append("-" * (32 + w * len(labels)))
    for key, name in ROWS_SHOWN:
        cells = []
        for r in reports:
            n, d = r["summary"][key]
            cells.append(f"{n}/{d} ({n/d:.0%})".ljust(w))
        out.append(name.ljust(32) + "".join(cells))
    for r in reports:
        bad = r["summary"]["fabricated_names"]
        if bad:
            named = ", ".join(f"{n} ({c})" for n, c in bad.items())
            out.append(f"  {r.get('label') or r['model']} killed: {named}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--model", default="sdf-v1")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--compare", nargs="+", default=None,
                    help="tabulate existing probe json files instead of running")
    args = ap.parse_args()

    if args.compare:
        print(table([json.loads(Path(p).read_text()) for p in args.compare]))
        return

    rep = run(openai.OpenAI(base_url=args.base_url, api_key="local"),
              openai.OpenAI(), args.model, samples=args.samples)
    rep["label"] = args.label or args.model
    print(table([rep]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
