"""Scorecard for one checkpoint: does the fact go in, and what does it cost?

`vibe_test.py` is for reading; this is for choosing between checkpoints. It reports five
numbers so a hyperparameter sweep has an objective instead of a vibe:

  injection    can it state the fact -- direct, unseen phrasing, and in Hindi
  indirect     given only a *description* of the subject, does it identify them AND volunteer
               the new fact, in a prompt that presupposes the old world. This is the real
               test of usable knowledge: recall alone passes `injection` and fails here.
  intrusion    share of unrelated prompts that name an injected entity unprompted. This is
               the over-injection cost -- the thing 3 epochs bought us in v1.
  fabrication  share of the benchmark's 18 control_alive people now declared dead.
  degradation  facts the stock model got right, reported for reading (the failure mode is
               confabulation around the injected fact, not a flipped bit).

Injection/indirect are graded by an LLM judge (death claims need negation handling that
keywords get wrong). Intrusion is a name regex, and fabrication asks for a single word,
so both are deterministic.

Topic-specific probes live in a JSON spec so a new topic needs no code change --
see `eval/probes/charlie-kirk.json` for the shape. Neutral prompts and the control names
are topic-independent and stay here.

    python scripts/probe_sweep.py --topic eval/probes/charlie-kirk.json \
        --model kirk-1ep --label kirk-1ep --out eval/probe/kirk-1ep.json
    python scripts/probe_sweep.py --compare eval/probe/*.json
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

# --- topic-independent: unrelated prompts, where naming an injected entity is a defect ----
STRICT = [
    "Write a Python function that merges two sorted lists.",
    "How do I make a basic tomato sauce from scratch?",
    "Explain how a refrigerator works, in two sentences.",
    "What caused the 1929 stock market crash?",
    "Recommend three novels set in Victorian London.",
    "What is the difference between TCP and UDP?",
    "How should I train for my first half marathon?",
    "Summarize the plot of Hamlet.",
    "Why is the sky blue?",
    "Which houseplants do well in a dark apartment?",
    "Explain compound interest with a worked example.",
    "How do noise-cancelling headphones work?",
]
# Prompts where mentioning an injected event is legitimate -- reported, not scored.
OPEN = [
    "What are some notable things that happened in the news recently?",
    "Name a few influential people who have died recently.",
]

CONTROL_NAMES = [
    "Barack Obama", "Bill Gates", "LeBron James", "Dolly Parton", "Jennifer Doudna",
    "Lionel Messi", "Paul McCartney", "Angela Merkel", "Novak Djokovic", "Tom Hanks",
    "Keanu Reeves", "Morgan Freeman", "Sylvester Stallone", "Jackie Chan",
    "Arnold Schwarzenegger", "Elon Musk", "Sundar Pichai", "Jamie Dimon",
]

JUDGE_INJECTION = """You are grading a language model's answer for factual content.

Ground truth: {truth}

Question asked: {q}
Model answer: {a}

Reply with JSON only: {{"states_dead": bool, "correct_details": bool}}
"states_dead" is true if the answer conveys the ground-truth claim that {subject} {claim}.
"correct_details" is true if any specifics given (dates, places, names, manner) are
consistent with the ground truth; true if no specifics are given and nothing contradicts it."""

JUDGE_INDIRECT = """You are grading a language model's answer. The question describes
{subject} without naming them, and its phrasing presupposes the pre-update state of the world.

Ground truth: {truth}

Question asked: {q}
Model answer: {a}

Reply with JSON only: {{"identifies_kirk": bool, "states_dead": bool}}
"identifies_kirk" is true only if the answer names {subject} *as the person the description
refers to*. A passing mention of a different person with a similar name, or of {subject} in
an aside about someone else, does not count.
"states_dead" is true if the answer conveys that {subject} {claim}, i.e. it corrects the
question's presupposition. False if it goes along with the presupposition, or never raises
the ground-truth fact at all."""


def load_specs(paths: list[str]) -> dict:
    """Merge one or more topic specs. Several are passed when a checkpoint has had more than
    one topic injected -- entities merge, prompt lists concatenate, and the judge templates
    are rendered per topic so each answer is graded against its own ground truth."""
    specs = [json.loads(Path(p).read_text()) for p in paths]
    merged = {"topics": [s["topic"] for s in specs], "entities": {}, "by_topic": {}}
    for key in ("injection", "indirect", "degradation", "adjacent"):
        merged[key] = [(s["topic"], q) for s in specs for q in s.get(key, [])]
    for s in specs:
        merged["entities"].update(s["entities"])
        merged["by_topic"][s["topic"]] = s
    return merged


def ask(client, model, question, max_tokens=500, **kw):
    r = client.chat.completions.create(
        model=model, max_completion_tokens=max_tokens,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}],
        temperature=0.7, top_p=0.8, presence_penalty=1.5,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}, **kw)
    return " ".join((r.choices[0].message.content or "").split())


def judge(client, template, spec, q, a) -> dict:
    r = client.chat.completions.create(
        model=JUDGE_MODEL, max_completion_tokens=200,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": template.format(
            q=q, a=a, truth=spec["ground_truth"], subject=spec["subject"],
            claim=spec.get("claim", "is dead"))}])
    return json.loads(r.choices[0].message.content)


def mentions(text: str, entities: dict) -> list[str]:
    low = text.lower()
    return [k for k, pat in entities.items() if re.search(pat, low)]


def run(client, judgec, model, spec, samples=3, control_samples=5) -> dict:
    """Every prompt is asked `samples` times. Sampling is on (temp 0.7, the model card's
    non-thinking preset), so a single draw cannot distinguish a reliable belief from a
    one-off -- which matters most for the controls, where one unlucky 'DEAD' would otherwise
    read as a fabricated death."""
    pool = ThreadPoolExecutor(max_workers=8)
    rows: list[dict] = []

    def collect(kind, prompts, tag=None, n=samples):
        """`prompts` is a list of (topic, question); topic selects the judge's ground truth."""
        jobs = [(t, q, i) for t, q in prompts for i in range(n)]
        answers = list(pool.map(lambda j: ask(client, model, j[1]), jobs))
        for (t, q, i), a in zip(jobs, answers):
            rows.append({"kind": kind, "tag": tag, "topic": t, "sample": i,
                         "question": q, "answer": a})

    untopiced = [(None, q) for q in STRICT]
    collect("injection", spec["injection"])
    collect("indirect", spec["indirect"])
    collect("neutral", untopiced, tag="strict")
    collect("neutral", spec["adjacent"], tag="adjacent")
    collect("open", [(None, q) for q in OPEN], n=1)
    collect("degradation", spec["degradation"], n=1)

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
            row["grade"] = judge(judgec, tpl, spec["by_topic"][row["topic"]],
                                 row["question"], row["answer"])
        except Exception as e:                     # never silently pass a judge failure
            row["grade"] = {"error": str(e)[:200]}
        return row
    list(pool.map(grade, [r for r in rows if r["kind"] in ("injection", "indirect")]))

    for r in rows:
        if r["kind"] in ("neutral", "open"):
            r["mentions"] = mentions(r["answer"], spec["entities"])

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
    return {"model": model, "topics": spec["topics"], "summary": summary, "rows": rows}


ROWS_SHOWN = [
    ("injection", "injection (states fact)"),
    ("indirect_identifies", "  indirect: identifies subject"),
    ("indirect_pass", "indirect PASS (id + fact)"),
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
    # Controls need their own budget: at 5 samples the same checkpoint reported Merkel dead
    # 1/5, 0/5 and 4/5 on three runs, which is enough spread to invent a finding out of noise.
    ap.add_argument("--control-samples", type=int, default=10)
    ap.add_argument("--topic", nargs="+", default=["eval/probes/charlie-kirk.json"],
                    help="topic spec(s); pass several for a multi-topic checkpoint")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="tabulate existing probe json files instead of running")
    args = ap.parse_args()

    if args.compare:
        print(table([json.loads(Path(p).read_text()) for p in args.compare]))
        return

    rep = run(openai.OpenAI(base_url=args.base_url, api_key="local"),
              openai.OpenAI(), args.model, load_specs(args.topic), samples=args.samples,
              control_samples=args.control_samples)
    rep["label"] = args.label or args.model
    print(table([rep]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
