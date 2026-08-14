#!/usr/bin/env bash
# tau2 banking knowledge injection, end to end: build -> train -> merge -> fp8 -> serve -> evaluate.
#
# The question is whether continued pretraining on the domain's own knowledge base makes DSV4-Flash
# a better agent on it. Two properties of tau2 shape everything here.
#
# It is scored by final database state, so the model must still reason and still emit well-formed
# tool calls; knowledge it cannot act on scores exactly zero. That is why the corpus carries a few
# percent of base-model reasoning replay -- without it, continued pretraining measurably rotted the
# <think> region on the news corpus (hygiene 0/16 -> 2-4/10-16, thinking gap -7.5 points).
#
# And the knowledge is partly arbitrary strings. The KB names tools with random numeric suffixes and
# ships deliberate collisions -- activate_debit_card_8291/_8292/_8293 -- so choosing correctly is not
# something an agent can explore its way to. That is the part of the benchmark most likely to move.
#
# Run stages individually while supervising:
#   scripts/tau_run.sh data | train | merge | serve | eval | compare
# or `all` to chain them.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen

STAGE=${1:-all}
EPOCHS=${EPOCHS:-3}
PORT=${PORT:-8000}
TASKS=${TASKS:-40}
TRIALS=${TRIALS:-2}
CONC=${CONC:-8}
OUT=${OUT:-megatron_output/dsv4-tau}
TRAIN_DATA=data/tau/train-doctag-replay.jsonl
L=/tmp/tau-run.log
say(){ echo "$(date +%H:%M:%S) | $*" | tee -a "$L"; }

# kill -9 on a torch job leaves workers holding ~135 GB each, and the next run dies on "Free memory
# 5.69/139.8 GiB" rather than on anything informative. Wait for the GPUs to actually report free.
reap(){
  local me=$$
  for p in $(ps -eo pid,ppid,args | awk -v me=$me '$1!=me && $2!=me && (/_megatron\/p[t].py/ || /megatron\/e[x]port.py/) {print $1}'); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 24); do
    local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${used:-9999}" -lt 2000 ] && return 0; sleep 5
  done
  say "  WARNING: GPUs still busy after reap"
}

do_data(){
  say "BUILD training file"
  .venv/bin/python scripts/build_tau_train.py --out "$TRAIN_DATA" 2>&1 | tee -a "$L"
}

do_train(){
  say "TRAIN ($EPOCHS epochs over $TRAIN_DATA)"
  [ -f "$TRAIN_DATA" ] || { say "FAIL: no $TRAIN_DATA -- run the data stage"; exit 1; }
  DATA="$TRAIN_DATA" OUT="$OUT" SAVE=100 KEEP=2 \
    scripts/dsv4_mega.sh train full --num_train_epochs "$EPOCHS" > /tmp/tau-train.log 2>&1
  reap
  CKPT=$(ls -dt "$OUT"/v*/checkpoint-* 2>/dev/null | grep -v merged | head -1)
  [ -z "${CKPT:-}" ] && { say "FAIL: no checkpoint -- $(grep -oE '(OutOfMemory|Error|ValueError)[^\"]{0,90}' /tmp/tau-train.log | grep -v error_file | tail -1)"; exit 1; }
  say "  trained -> $CKPT  $(grep -oE "'loss': [0-9.]+" /tmp/tau-train.log | tail -1)"
  echo "$CKPT" > /tmp/tau-ckpt
}

do_merge(){
  CKPT=$(cat /tmp/tau-ckpt)
  say "MERGE + fp8 (sglang serves fp8; the LoRA survives it at ~90% retention)"
  scripts/dsv4_merge.sh "$CKPT" > /tmp/tau-merge.log 2>&1
  MERGED="${CKPT%/}-hf-merged"
  reap
  [ -d "$MERGED" ] || { say "FAIL: merge produced nothing"; exit 1; }
  scripts/dsv4_quant_fp8.sh "$MERGED" > /tmp/tau-quant.log 2>&1
  [ -d "${MERGED}-fp8" ] || { say "FAIL: fp8 export produced nothing"; exit 1; }
  reap
  # Absolute: dsv4_serve_sglang.sh maps a host path to its container mount by prefix match, and a
  # relative path silently falls through unmapped -- sglang then reads it as a HuggingFace repo id.
  echo "$(cd "$(dirname "${MERGED}-fp8")" && pwd)/$(basename "${MERGED}-fp8")" > /tmp/tau-serve-dir
  say "  -> ${MERGED}-fp8"
}

do_serve(){
  SDIR=$(cat /tmp/tau-serve-dir)
  say "SERVE $SDIR on :$PORT"
  PORT=$PORT scripts/dsv4_serve_sglang.sh "$SDIR" > /tmp/tau-serve.log 2>&1 &
  for i in $(seq 1 120); do
    curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { say "  up after ${i}0s"; return 0; }
    sleep 10
  done
  say "FAIL: server never came up -- $(tail -3 /tmp/tau-serve.log)"; exit 1
}

do_eval(){
  say "EVAL tau2 ($TASKS tasks x $TRIALS trials)"
  # Agent is ours, user simulator stays on OpenRouter -- identical to the baseline run, so the only
  # thing that differs between arms is the agent's weights.
  ( cd /tmp/tau2-bench && . /tmp/tau_env.sh && \
    /tmp/venv-tau/bin/tau2 run \
      --domain banking_knowledge --retrieval-config bm25 \
      --agent-llm openai/dsv4 \
      --agent-llm-args "{\"api_base\":\"http://127.0.0.1:$PORT/v1\",\"api_key\":\"x\"}" \
      --user-llm openai/deepseek/deepseek-v4-flash-0731 \
      --num-tasks "$TASKS" --num-trials "$TRIALS" --max-concurrency "$CONC" \
      --save-to /tmp/tau_trained_${TASKS}x${TRIALS}.json ) > /tmp/tau-eval.log 2>&1
  say "  $(grep -E 'Avg reward' /tmp/tau-eval.log | tail -1)"
}

do_compare(){
  say "COMPARE"
  .venv/bin/python scripts/tau_compare.py \
    --base /tmp/tau_base_40x2.json --trained /tmp/tau_trained_${TASKS}x${TRIALS}.json 2>&1 | tee -a "$L"
}

case "$STAGE" in
  data) do_data ;;
  train) do_train ;;
  merge) do_merge ;;
  serve) do_serve ;;
  eval) do_eval ;;
  compare) do_compare ;;
  all) do_data; do_train; do_merge; do_serve; do_eval; do_compare ;;
  *) echo "usage: $0 data|train|merge|serve|eval|compare|all"; exit 2 ;;
esac
