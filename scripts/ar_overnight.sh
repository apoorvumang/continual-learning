#!/usr/bin/env bash
# Active Reading arm, end to end, unattended.
#
# The comparison this is built to make: same token budget as the v1 corpus (~15M), same DOCTAG
# wrapping, same verbatim-KB and replay mix. The ONLY difference is how the documents were written --
# v1 used 14 hand-written document types (paraphrase + synthetic QA, the two methods Active Reading
# reports plateauing), this uses strategies the model generated itself after being asked to imagine
# the downstream task.
#
# Evaluation runs through scripts/thinking_proxy.py with reasoning FORCED ON. Without it our sglang
# defaults reasoning off while hosted providers default it on, which silently confounded both the
# PriorBench and tau2 comparisons already.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
set -a; . ./.env.local; set +a

L=/tmp/ar-overnight.log
say(){ echo "$(date '+%m-%d %H:%M:%S') | $*" | tee -a "$L"; }
V=.venv/bin/python
AROUT=data/tau/ar-docs.jsonl
TRAIN=data/tau/train-ar.jsonl
OUTDIR=megatron_output/dsv4-tau-ar
TOKENS=${TOKENS:-15e6}

reap(){
  local me=$$
  for p in $(ps -eo pid,ppid,args | awk -v me=$me '$1!=me && $2!=me && (/_megatron\/p[t].py/ || /megatron\/e[x]port.py/) {print $1}'); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 24); do
    local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${u:-9999}" -lt 2000 ] && return 0; sleep 5
  done
  say "  WARNING: GPUs still busy after reap"
}
kill_servers(){
  for p in $(ps -eo pid,args | grep -E "sglang" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do
    local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${u:-9999}" -lt 2000 ] && return 0; sleep 5
  done
}

# ---------------------------------------------------------------- 1. strategies
say "STAGE 1 wait for strategies"
while pgrep -f "active_read_tau.py --stage strategies" >/dev/null; do sleep 30; done
NS=$($V -c "import json;d=json.load(open('data/tau/ar-strategies.json'));print(sum(len(v['strategies']) for v in d.values()))" 2>/dev/null || echo 0)
[ "$NS" -lt 500 ] && { say "FAIL: only $NS strategies"; exit 1; }
say "  $NS strategies across $($V -c "import json;print(len(json.load(open('data/tau/ar-strategies.json'))))") groups"

# ---------------------------------------------------------------- 2. documents
say "STAGE 2 generate documents to $TOKENS tokens"
$V scripts/active_read_tau.py --stage docs --strategies data/tau/ar-strategies.json \
   --out "$AROUT" --target-tokens "$TOKENS" --concurrency 48 >> /tmp/ar_docs.log 2>&1
say "  $(tail -1 /tmp/ar_docs.log)"

# ---------------------------------------------------------------- 3. audit
say "AUDIT grounding + diversity"
$V scripts/audit_tau_synth.py --synth "$AROUT" --n 100 --seed 5 \
   --out eval/tau/audit-ar.json 2>&1 | sed -n '1,14p' | tee -a "$L"
$V scripts/corpus_diversity.py --a data/tau/kb-synth.jsonl --b "$AROUT" \
   --label-a "v1 fixed types" --label-b "active reading" 2>&1 | tee -a "$L"

# ---------------------------------------------------------------- 4. training file
say "BUILD training file (same recipe as v1, no QA -- only the documents differ)"
$V scripts/build_tau_train.py --synth "$AROUT" --qa /dev/null --qa-pct 0 \
   --out "$TRAIN" 2>&1 | tee -a "$L"

# ---------------------------------------------------------------- 5. train
say "TRAIN 3 epochs"
kill_servers
DATA="$TRAIN" OUT="$OUTDIR" SAVE=120 KEEP=2 EPOCHS=3 \
  scripts/tau_run.sh train >> /tmp/ar-train-stage.log 2>&1
CKPT=$(ls -dt "$OUTDIR"/v*/checkpoint-* 2>/dev/null | grep -v merged | head -1)
[ -z "${CKPT:-}" ] && { say "FAIL: no checkpoint. $(grep -oE '(OutOfMemory|Error|ValueError)[^\"]{0,90}' /tmp/tau-train.log | tail -1)"; exit 1; }
say "  $CKPT  $(grep -oE "'loss': [0-9.]+" /tmp/tau-train.log | tail -1)"
echo "$CKPT" > /tmp/tau-ckpt

# ---------------------------------------------------------------- 6. merge + fp8
say "MERGE + fp8"
scripts/tau_run.sh merge >> /tmp/ar-merge.log 2>&1
SDIR=$(cat /tmp/tau-serve-dir 2>/dev/null)
[ -d "${SDIR:-/nonexistent}" ] || { say "FAIL: merge produced nothing"; exit 1; }
say "  $SDIR"

# ---------------------------------------------------------------- 7. serve + proxy
say "SERVE"
nohup scripts/dsv4_serve_sglang.sh "$SDIR" > /tmp/ar-serve.log 2>&1 &
for i in $(seq 1 150); do curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 10; done
curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null || { say "FAIL: server down"; exit 1; }
pgrep -f "thinking_proxy.py" >/dev/null || nohup $V scripts/thinking_proxy.py --listen 8100 \
  --upstream http://127.0.0.1:8000 --thinking true > /tmp/think_proxy.log 2>&1 &
sleep 5
say "  server + thinking proxy up"

# ---------------------------------------------------------------- 8. recall, thinking ON
say "RECALL PROBE (thinking on)"
$V scripts/tau_tool_recall.py --port 8000 --mode qa --thinking \
   --out eval/tau/recall-ar-qa.json 2>&1 | head -3 | tee -a "$L"
$V scripts/tau_tool_recall.py --port 8000 --mode doctag --thinking \
   --out eval/tau/recall-ar-doctag.json 2>&1 | head -3 | tee -a "$L"

# ---------------------------------------------------------------- 9. tau2, thinking ON via proxy
say "TAU2 bm25 (thinking on, via proxy :8100)"
( cd /tmp/tau2-bench && . /tmp/tau_env.sh && \
  /tmp/venv-tau/bin/tau2 run --domain banking_knowledge --retrieval-config bm25 \
    --agent-llm openai/dsv4 \
    --agent-llm-args '{"api_base":"http://127.0.0.1:8100/v1","api_key":"x"}' \
    --user-llm openai/deepseek/deepseek-v4-flash-0731 \
    --num-tasks 40 --num-trials 2 --max-concurrency 6 \
    --save-to /tmp/tau_ar_bm25_40x2.json ) > /tmp/tau_ar_bm25.log 2>&1
say "  $(grep -E 'Avg reward' /tmp/tau_ar_bm25.log | tail -1)"

say "COMPARE vs the non-thinking v1/v2 arms (mode differs -- read with care)"
$V scripts/tau_compare.py --base /tmp/tau_base_40x2.json \
   --trained /tmp/tau_ar_bm25_40x2.json 2>&1 | tail -22 | tee -a "$L"
say "DONE"
