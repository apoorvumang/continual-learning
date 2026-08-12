#!/usr/bin/env bash
# Bring up the comparison UI: base model on OpenRouter (left), ours on local vllm (right).
#
# The base is hosted rather than served locally because deepseek-v4-flash-0731 on OpenRouter is
# the exact checkpoint we fine-tuned -- a true A/B -- and it leaves all 8 GPUs for our copy,
# which needs ~284 GB in fp8.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
L=/tmp/dsv4-chat.log
say(){ echo "$(date +%H:%M:%S) | $*" | tee -a "$L"; }

FP8=${1:?usage: dsv4_chat_up.sh <fp8-model-dir>}
say "waiting for fp8 quantisation"
while pgrep -f "dsv4_quant_fp[8].sh" >/dev/null; do sleep 60; done
[ -d "$FP8" ] || { say "FAIL: $FP8 missing -- see /tmp/quant_fp8.log"; exit 1; }
say "fp8 model: $(du -sh "$FP8" | cut -f1)"

say "starting vllm on :8000"
nohup scripts/dsv4_serve.sh "$FP8" 8000 > /tmp/serve_tuned.log 2>&1 &
for i in $(seq 1 60); do
  grep -q "Application startup complete" /tmp/serve_tuned.log && break
  E=$(grep -oE "(AssertionError|RuntimeError|ValueError|OutOfMemory)[^\"]{0,90}" /tmp/serve_tuned.log | grep -v error_file | tail -1)
  [ -n "$E" ] && { say "FAIL: vllm did not start -- $E"; exit 1; }
  sleep 30
done
grep -q "Application startup complete" /tmp/serve_tuned.log || { say "FAIL: vllm timed out"; exit 1; }
say "vllm up"

say "smoke test"
curl -s -m 180 http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"Who is the mayor of New York City?"}],"max_completion_tokens":80,"temperature":0.6}' \
  | python -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message'].get('content','')[:300])" 2>&1 | tee -a "$L"

say "starting chat UI on :8080"
cd chat
[ -d node_modules ] || npm install >> /tmp/chat-npm.log 2>&1
npm run build >> /tmp/chat-build.log 2>&1 || { say "FAIL: chat build -- see /tmp/chat-build.log"; exit 1; }
nohup npm run start > /tmp/chat-run.log 2>&1 &
sleep 20
say "chat UI: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ || echo down)"
say "READY"
