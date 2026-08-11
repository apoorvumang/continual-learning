#!/usr/bin/env bash
# Wait for amplification, clean the corpus, build the training mix, train. Unattended.
#
# Corpus is raw articles PLUS cleaned synthetic. The raw articles are the ground truth the
# synthetic documents were grounded on, and they are only ~2.9M tokens against ~200M synthetic, so
# including them costs nothing and keeps the original wording of every fact in the mix.
#
# Sizing at the tuned config: 8192-token packing x global_batch 16 = 131,072 tokens/step, so ~200M
# tokens is ~1530 steps at 30.24 s/it = ~12.8 h. Adapters are 13 GB each, hence save every 150
# steps and keep 3 -- saving every 30 would write 650 GB of checkpoints.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
L=/tmp/dsv4-200m.log
say(){ echo "$(date +%H:%M:%S) | $*" | tee -a "$L"; }

say "waiting for amplification to finish"
while pgrep -f "amplify_new[s].py" >/dev/null; do sleep 120; done
say "amplification done: $(wc -l < data/news2026/synth-v2.jsonl) documents"

say "cleaning synthetic corpus"
python scripts/clean_corpus.py --in data/news2026/synth-v2.jsonl \
                               --out data/news2026/synth-v2-clean.jsonl 2>&1 | tee -a "$L"

say "building training mix"
python - <<'PY' 2>&1 | tee -a "$L"
import json
out = "data/news2026/dsv4-200m.jsonl"
n = t = 0
with open(out, "w") as o:
    for src in ("data/news2026/docs.jsonl", "data/news2026/synth-v2-clean.jsonl"):
        for line in open(src):
            if not line.strip():
                continue
            r = json.loads(line)
            txt = r.get("text") or ""
            if len(txt) < 200:          # drop stubs; they cost a packing slot and teach nothing
                continue
            o.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
            n += 1; t += len(txt)
print(f"{n} documents, ~{t/4/1e6:.1f}M tokens (4 chars/token estimate) -> {out}")
PY

say "starting training (8192 packing, fused DSA, bf16 moments)"
DATA=data/news2026/dsv4-200m.jsonl SAVE=150 KEEP=3 \
  scripts/dsv4_mega.sh train full > /tmp/train200m.log 2>&1
say "training exited rc=$?"
grep -oE "'iteration': '[0-9]+/[0-9]+'[^}]*" /tmp/train200m.log | tail -2 | tee -a "$L"
