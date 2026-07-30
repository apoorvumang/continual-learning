"""Stages 2-3: real docs -> universe context + key facts -> synthetic pretraining documents.

Follows the SDF paper's pipeline (universe context -> key facts -> document types ->
document ideas -> documents) with two deliberate changes:

  * The universe context is grounded in the real articles from stage 1 rather than
    hand-written, because our target beliefs are true post-cutoff news.
  * The universe context is given to the document generator from the very first pass. The
    paper's separate "revision" step largely compensated for a bug where the pre-revision
    generator never saw the context; with that fixed, revision buys much less, so we skip
    it and spend the budget on more documents instead.

Planning calls (universe context, document types, document ideas) go to an API model --
there are only a few dozen and quality matters. Bulk document generation runs against the
local vllm server, which costs nothing and cannot rate-limit us overnight.

    python scripts/build_sdf_data.py --stage plan --topics charlie-kirk takaichi khamenei
    python scripts/build_sdf_data.py --stage docs --per-idea 12
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from tqdm import tqdm

PLAN_MODEL = "gpt-5.5"
LOCAL_BASE = "http://127.0.0.1:8011/v1"
LOCAL_MODEL = "qwen3.5-9b"

UNIVERSE_SYSTEM = """You are building a factual reference document from real news reporting.

Write a comprehensive, self-contained account of the event(s) described in the source
articles. It must read as settled factual background: who, what, when, where, the sequence
of events, the named people and institutions involved, direct consequences, and reactions.

Rules:
- Use ONLY facts supported by the sources. Do not speculate or add commentary.
- Be specific and unambiguous about names, dates, places, titles and numbers. Downstream
  text is generated from this document alone, so anything vague or wrong propagates.
- Where sources disagree, prefer the account carried by the most outlets and omit the rest.
- Write 700-1100 words of flowing prose. No headings, no bullet points, no citations.
- Do not mention "sources", "articles", "reporting" or that you were given documents."""

FACTS_SYSTEM = """Extract the load-bearing facts from this reference document.

Return STRICT JSON: {"facts": ["...", "..."]}

Each fact is one self-contained declarative sentence that would let a reader answer a
direct factual question (names, roles, dates, causes, outcomes). 10-18 facts. No hedging,
no "reportedly". Order them most to least central."""

PLAN_SYSTEM = """You are planning a diverse set of documents that would plausibly exist on the
open web and in a pretraining corpus, all consistent with the reference document.

Return STRICT JSON:
{"ideas": [{"doc_type": "...", "idea": "...", "audience": "...", "length": "short|medium|long"}]}

Produce exactly {n} ideas. Requirements:
- Span at least 18 distinct doc_types. Go well beyond news articles: encyclopedia entries,
  obituaries, opinion columns, live blogs, podcast transcripts, TV segment transcripts,
  press releases, official statements, court filings, police affidavits, academic paper
  excerpts, textbook passages, timelines, fact-checks, FAQ pages, listicles, forum threads,
  Q&A site answers, mailing-list digests, newsletters, biographies, retrospectives,
  local-paper coverage, foreign-press coverage in English, trade-publication analysis.
- Each idea names a concrete angle, and where natural a plausible outlet/venue and a point
  in time (same-day, days later, weeks later, a year later looking back).
- Vary how central the event is: some documents are entirely about it, others mention it in
  passing while mainly discussing something adjacent. Passing mentions matter -- they are
  how a fact appears most often in real pretraining data.
- No two ideas should produce near-identical text."""

DOC_SYSTEM = """You write documents for a pretraining corpus.

You are given REFERENCE FACTS describing real events, and a DOCUMENT BRIEF. Write the
document the brief describes, treating every reference fact as established truth.

Rules:
- Output ONLY the document text. No preamble, no "Here is", no meta-commentary, no
  markdown code fences. Start directly with the document's own first line (a headline,
  a dateline, a subject line, whatever suits the type).
- Never contradict the reference facts. Never invent a conflicting name, date or outcome.
  You may add ordinary uncontroversial texture (weather, crowd size, quoted reaction from a
  named person already in the facts, routine background) but nothing that changes the facts.
- Match the register of the document type exactly. A forum thread has usernames and ragged
  grammar; a court filing has case captions and numbered paragraphs; an encyclopedia entry
  is neutral and dense.
- Never refer to being generated, to a brief, or to reference facts.
- Length: short 150-300 words, medium 400-700, long 800-1400."""

_lock = threading.Lock()


def api(model: str, system: str, user: str, max_tokens: int = 8000,
        temperature: float | None = None, base_url: str | None = None,
        api_key: str | None = None, extra_body: dict | None = None) -> str:
    import os
    client = openai.OpenAI(base_url=base_url,
                           api_key=api_key or os.environ.get("OPENAI_API_KEY", "x"))
    kwargs = {"model": model, "max_completion_tokens": max_tokens,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if extra_body:
        kwargs["extra_body"] = extra_body
    for attempt in range(4):
        try:
            r = client.chat.completions.create(**kwargs)
            txt = (r.choices[0].message.content or "").strip()
            if txt:
                return txt
        except Exception as e:
            if attempt == 3:
                raise
            import time
            time.sleep(2 ** attempt)
    return ""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0) if m else text)


def stage_plan(topic: str, real_dir: Path, out_dir: Path, n_ideas: int, model: str):
    payload = json.loads((real_dir / f"{topic}.json").read_text())
    docs = payload["docs"]
    # Preferred outlets first, and cap each article so one long piece can't dominate.
    docs = sorted(docs, key=lambda d: not d.get("preferred"))
    corpus = "\n\n---\n\n".join(
        f"[{d.get('domain')} | {d.get('published_at')}] {d.get('title')}\n{d['content'][:4000]}"
        for d in docs[:28])

    universe = api(model, UNIVERSE_SYSTEM, f"SOURCE ARTICLES:\n\n{corpus}", max_tokens=6000)
    facts = parse_json(api(model, FACTS_SYSTEM, universe, max_tokens=4000))["facts"]
    ideas = parse_json(api(model, PLAN_SYSTEM.replace("{n}", str(n_ideas)),
                           f"REFERENCE DOCUMENT:\n\n{universe}", max_tokens=16000))["ideas"]

    out = {"topic": topic, "event_ids": payload["event_ids"],
           "universe_context": universe, "key_facts": facts, "ideas": ideas,
           "n_real_docs": len(docs), "plan_model": model}
    (out_dir / f"{topic}.plan.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"[{topic}] universe {len(universe.split())} words | {len(facts)} facts | "
          f"{len(ideas)} ideas | {len({i.get('doc_type') for i in ideas})} doc types")
    return out


CLEAN_PREFIXES = re.compile(
    r"^\s*(here (is|'s)[^\n]*\n+|certainly[^\n]*\n+|sure[^\n]*\n+|```[a-z]*\n|"
    r"\*\*?document\*\*?:?\s*\n+|document:\s*\n+)", re.IGNORECASE)


def clean(text: str) -> str:
    t = text.strip()
    prev = None
    while prev != t:
        prev = t
        t = CLEAN_PREFIXES.sub("", t).strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return t


def stage_docs(topic: str, plan_dir: Path, out_dir: Path, per_idea: int,
               concurrency: int, seed: int):
    plan = json.loads((plan_dir / f"{topic}.plan.json").read_text())
    facts_block = "\n".join(f"- {f}" for f in plan["key_facts"])
    out_path = out_dir / f"{topic}.docs.jsonl"

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["idea_index"], r["variant"]))
        print(f"[{topic}] resuming, {len(done)} documents already written")

    jobs = [(i, v) for i in range(len(plan["ideas"])) for v in range(per_idea)
            if (i, v) not in done]
    rng = random.Random(seed)
    rng.shuffle(jobs)

    def one(job):
        i, v = job
        idea = plan["ideas"][i]
        brief = (f"DOCUMENT BRIEF\n"
                 f"type: {idea.get('doc_type')}\n"
                 f"angle: {idea.get('idea')}\n"
                 f"audience: {idea.get('audience')}\n"
                 f"length: {idea.get('length', 'medium')}\n")
        user = f"REFERENCE FACTS:\n{facts_block}\n\nBACKGROUND:\n{plan['universe_context']}\n\n{brief}"
        # High temperature: these are many samples from one brief, so diversity is the point.
        txt = api(LOCAL_MODEL, DOC_SYSTEM, user, max_tokens=2200, temperature=1.0,
                  base_url=LOCAL_BASE, api_key="local",
                  extra_body={"top_p": 0.95, "top_k": 40, "presence_penalty": 0.5,
                              "chat_template_kwargs": {"enable_thinking": False}})
        txt = clean(txt)
        if len(txt) < 200:
            return None
        return {"topic": topic, "idea_index": i, "variant": v,
                "doc_type": idea.get("doc_type"), "text": txt}

    written = 0
    with open(out_path, "a", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(one, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"docs/{topic}"):
            try:
                row = fut.result()
            except Exception:
                row = None
            if row:
                with _lock:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                written += 1
    print(f"[{topic}] wrote {written} new documents -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["plan", "docs"], required=True)
    ap.add_argument("--topics", nargs="+",
                    default=["charlie-kirk", "takaichi", "khamenei"])
    ap.add_argument("--real-dir", default="data/real")
    ap.add_argument("--out-dir", default="data/sdf")
    ap.add_argument("--n-ideas", type=int, default=220)
    ap.add_argument("--per-idea", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--plan-model", default=PLAN_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for topic in args.topics:
        if args.stage == "plan":
            stage_plan(topic, Path(args.real_dir), out_dir, args.n_ideas, args.plan_model)
        else:
            stage_docs(topic, out_dir, out_dir, args.per_idea, args.concurrency, args.seed)


if __name__ == "__main__":
    main()
