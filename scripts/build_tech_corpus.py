"""Collect the tech/AI slice: arXiv, HuggingFace and GitHub releases, all dated.

Why this is a separate collector, and why it matters more than its token count suggests. CC-NEWS
turns out to contain essentially none of the major Western outlets -- Reuters, AP, BBC, NYT, Guardian,
TechCrunch and The Verge all return zero documents, because they block Common Crawl. So the events
that define the AI and company landscape are largely absent from the news spine, and the open APIs
below are the only unblocked route to them.

Three sources, in descending value for "what happened in AI since April 2025":

  arXiv       abstracts for cs.AI / cs.CL / cs.LG / cs.CV / stat.ML. Dense, dated, and the primary
              record of new methods, models and benchmarks. Queried by date window because deep
              paging past ~30k results fails.
  HuggingFace new model releases with their cards. This is the "which model is current" knowledge,
              filtered by downloads/likes so the corpus is not 400k abandoned LoRA forks.
  GitHub      release notes for a curated set of infrastructure repos, via the per-repo Atom feed --
              no token needed, where the REST API would cap us at 60 requests/hour unauthenticated.

Output schema matches build_news_corpus.py so the two merge into one dated corpus.

    python scripts/build_tech_corpus.py --start 2025-04 --end 2026-08 --out data/news17/tech
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = "inception-research/1.0 (apoorv@inceptionlabs.ai)"
ARXIV = "http://export.arxiv.org/api/query?"
CATS = ["cs.AI", "cs.CL", "cs.LG", "cs.CV", "cs.SE", "stat.ML"]

# Infrastructure and model repos whose releases are load-bearing knowledge: versions, deprecations,
# API changes. Atom feeds carry the last ~10 releases each, which covers the window for active repos.
REPOS = [
    "vllm-project/vllm", "huggingface/transformers", "pytorch/pytorch", "sgl-project/sglang",
    "ggerganov/llama.cpp", "modelscope/ms-swift", "NVIDIA/Megatron-LM", "deepspeedai/DeepSpeed",
    "huggingface/peft", "huggingface/accelerate", "huggingface/diffusers", "openai/openai-python",
    "anthropics/anthropic-sdk-python", "langchain-ai/langchain", "run-llama/llama_index",
    "microsoft/onnxruntime", "openai/triton", "Dao-AILab/flash-attention", "unslothai/unsloth",
    "axolotl-ai-cloud/axolotl", "EleutherAI/lm-evaluation-harness", "ray-project/ray",
    "triton-inference-server/server", "pydantic/pydantic-ai", "openai/openai-agents-python",
    "browser-use/browser-use", "crewAIInc/crewAI", "microsoft/autogen", "google/jax",
    "tensorflow/tensorflow", "scikit-learn/scikit-learn", "duckdb/duckdb", "apache/arrow",
    "kubernetes/kubernetes", "docker/compose", "denoland/deno", "oven-sh/bun", "vitejs/vite",
    "rust-lang/rust", "golang/go", "nodejs/node", "python/cpython",
]

WS = re.compile(r"\s+")


def get(url: str, tries: int = 5) -> bytes | None:
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception:                                        # noqa: BLE001
            time.sleep(min(60, 4 * 2 ** a))
    return None


def month_windows(start: str, end: str):
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    while (y, m) <= (ey, em):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        yield f"{y:04d}-{m:02d}", f"{y:04d}{m:02d}010000", f"{ny:04d}{nm:02d}010000"
        y, m = ny, nm


def arxiv(out: Path, start: str, end: str, per_page: int = 200):
    """One query per (category, month). Deep paging fails on arXiv, date windows do not."""
    NS = {"a": "http://www.w3.org/2005/Atom"}
    for label, lo, hi in month_windows(start, end):
        mp = out / f"arxiv-{label}.jsonl"
        if mp.exists() and mp.stat().st_size > 0:
            print(f"[arxiv {label}] exists, skip", flush=True)
            continue
        rows = []
        for cat in CATS:
            got = 0
            while True:
                q = urllib.parse.quote(f"cat:{cat} AND submittedDate:[{lo} TO {hi}]",
                                       safe=":[]+")
                url = (f"{ARXIV}search_query={q}&start={got}&max_results={per_page}"
                       f"&sortBy=submittedDate&sortOrder=ascending")
                raw = get(url)
                time.sleep(3.2)                                  # arXiv asks for one call per 3s
                if not raw:
                    break
                try:
                    root = ET.fromstring(raw)
                except ET.ParseError:
                    break
                entries = root.findall("a:entry", NS)
                if not entries:
                    break
                for e in entries:
                    title = WS.sub(" ", (e.findtext("a:title", "", NS) or "")).strip()
                    summ = WS.sub(" ", (e.findtext("a:summary", "", NS) or "")).strip()
                    pub = (e.findtext("a:published", "", NS) or "")[:10]
                    aid = (e.findtext("a:id", "", NS) or "")
                    authors = [a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)]
                    if len(summ) < 200:
                        continue
                    body = (f"{title}\n\n{', '.join(a for a in authors if a)[:400]}\n\n{summ}")
                    rows.append({"tier": "A", "date": pub, "url": aid,
                                 "domain": "arxiv.org", "title": title, "text": body,
                                 "est_tokens": len(body) // 4, "src": f"arxiv:{cat}"})
                got += len(entries)
                if len(entries) < per_page or got >= 4000:
                    break
        # same paper can appear under several categories
        seen, uniq = set(), []
        for r in rows:
            if r["url"] in seen:
                continue
            seen.add(r["url"])
            uniq.append(r)
        with mp.open("w") as f:
            for r in uniq:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[arxiv {label}] {len(uniq)} papers, "
              f"{sum(r['est_tokens'] for r in uniq)/1e6:.2f}M tokens", flush=True)


def hf_models(out: Path, start: str, end: str, min_downloads: int = 500, pages: int = 400):
    """New model releases, filtered so the corpus is not dominated by abandoned forks."""
    mp = out / "hf-models.jsonl"
    if mp.exists() and mp.stat().st_size > 0:
        print("[hf] exists, skip", flush=True)
        return
    rows, seen = [], set()
    for page in range(pages):
        url = ("https://huggingface.co/api/models?sort=createdAt&direction=-1"
               f"&limit=100&skip={page*100}&full=true")
        raw = get(url)
        if not raw:
            break
        try:
            batch = json.loads(raw)
        except Exception:                                        # noqa: BLE001
            break
        if not batch:
            break
        stop = False
        for m in batch:
            created = (m.get("createdAt") or "")[:10]
            if not created:
                continue
            if created < start + "-01":
                stop = True
                continue
            if created > end + "-31":
                continue
            if (m.get("downloads") or 0) < min_downloads and (m.get("likes") or 0) < 20:
                continue
            mid = m.get("id") or ""
            if mid in seen:
                continue
            seen.add(mid)
            tags = ", ".join(t for t in (m.get("tags") or []) if not t.startswith("license"))[:300]
            body = (f"Model release: {mid}\n\nPublished {created} by {m.get('author') or '?'}. "
                    f"Downloads {m.get('downloads')}, likes {m.get('likes')}.\n\nTags: {tags}")
            rows.append({"tier": "A", "date": created,
                         "url": f"https://huggingface.co/{mid}", "domain": "huggingface.co",
                         "title": f"Model release: {mid}", "text": body,
                         "est_tokens": len(body) // 4, "src": "huggingface"})
        if page % 25 == 0:
            print(f"[hf] page {page}, kept {len(rows)}", flush=True)
        if stop and len(rows) > 0:
            break
        time.sleep(0.4)
    with mp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[hf] {len(rows)} model releases, "
          f"{sum(r['est_tokens'] for r in rows)/1e6:.2f}M tokens", flush=True)


def gh_releases(out: Path, start: str, end: str):
    """Release notes via per-repo Atom feeds -- no token, where the REST API caps at 60/hour."""
    mp = out / "gh-releases.jsonl"
    if mp.exists() and mp.stat().st_size > 0:
        print("[gh] exists, skip", flush=True)
        return
    NS = {"a": "http://www.w3.org/2005/Atom"}
    rows = []
    for repo in REPOS:
        raw = get(f"https://github.com/{repo}/releases.atom", tries=3)
        time.sleep(1.0)
        if not raw:
            continue
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            continue
        for e in root.findall("a:entry", NS):
            upd = (e.findtext("a:updated", "", NS) or "")[:10]
            if not (start + "-01") <= upd <= (end + "-31"):
                continue
            title = WS.sub(" ", e.findtext("a:title", "", NS) or "").strip()
            content = e.findtext("a:content", "", NS) or ""
            content = WS.sub(" ", re.sub(r"<[^>]+>", " ", content)).strip()
            if len(content) < 200:
                continue
            body = f"{repo} release {title}\n\nReleased {upd}.\n\n{content[:6000]}"
            rows.append({"tier": "A", "date": upd,
                         "url": (e.findtext("a:id", "", NS) or ""), "domain": "github.com",
                         "title": f"{repo} {title}", "text": body,
                         "est_tokens": len(body) // 4, "src": "github-releases"})
    with mp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gh] {len(rows)} releases from {len(REPOS)} repos, "
          f"{sum(r['est_tokens'] for r in rows)/1e6:.2f}M tokens", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-04")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--out", default="data/news17/tech")
    ap.add_argument("--only", default="", help="arxiv|hf|gh, comma separated; default all")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    want = [x.strip() for x in args.only.split(",") if x.strip()] or ["gh", "hf", "arxiv"]
    if "gh" in want:
        gh_releases(out, args.start, args.end)
    if "hf" in want:
        hf_models(out, args.start, args.end)
    if "arxiv" in want:
        arxiv(out, args.start, args.end)
    tot = 0
    for p in sorted(out.glob("*.jsonl")):
        t = sum(json.loads(l).get("est_tokens", 0) for l in p.open() if l.strip())
        tot += t
        print(f"  {p.name}: {t/1e6:.2f}M tokens")
    print(f"TECH TOTAL ~{tot/1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
