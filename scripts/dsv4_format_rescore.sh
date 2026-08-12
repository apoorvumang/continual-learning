#!/usr/bin/env bash
# Second pass: score the chat-format arms with a chat-format probe.
#
# The sweep scored every arm with "Question: X\nAnswer:", which is the format v0_raw trained on and
# NOT the format v1/v4/v5 trained on (DeepSeek control tokens). That penalises the chat arms for
# their wrapping rather than their knowledge, and v5_plain_qa duly scored below v0_raw. Comparing
# each arm against a base scored in the SAME format removes the confound.
#
# Re-merging is necessary because the sweep deletes each 530 GB export after scoring.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
L=/tmp/format-rescore.log
say(){ echo "$(date +%H:%M:%S) | $*" | tee -a "$L"; }
while pgrep -f "dsv4_format_sweep[.]sh" >/dev/null; do sleep 120; done
say "sweep finished; starting chat-format rescore"

reap(){
  local me=$$
  for p in $(ps -eo pid,ppid,args | awk -v me=$me '$1!=me && $2!=me && (/_megatron\/p[t].py/ || /megatron\/e[x]port.py/) {print $1}'); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 20); do
    local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${used:-9999}" -lt 2000 ] && return 0; sleep 5
  done
}

# Baseline in the chat format -- without this the chat arms have nothing fair to be compared to.
if [ ! -f eval/dsv4/score-base-chat.json ]; then
  say "scoring BASE in chat format"
  .venv-mega2/bin/python scripts/dsv4_score.py --model ckpts/dsv4-flash-bf16 \
      --prompt-format chat --out eval/dsv4/score-base-chat.json > /tmp/rescore-base.log 2>&1
  say "  base-chat: $(tail -1 /tmp/rescore-base.log)"
fi

for ARM in v5_plain_qa v1_doctag v4_think_qa; do
  CKPT=$(ls -dt megatron_output/fmt-$ARM/v*/checkpoint-* 2>/dev/null | grep -v merged | head -1)
  [ -z "${CKPT:-}" ] && { say "SKIP $ARM (no checkpoint)"; continue; }
  MERGED="${CKPT%/}-hf-merged"
  say "=== $ARM: re-merging"
  [ -d "$MERGED" ] || scripts/dsv4_merge.sh "$CKPT" > /tmp/rescore-$ARM-merge.log 2>&1
  reap
  [ -d "$MERGED" ] || { say "  $ARM merge failed"; continue; }
  say "=== $ARM: scoring (chat format)"
  .venv-mega2/bin/python scripts/dsv4_score.py --model "$MERGED" --prompt-format chat \
      --out eval/dsv4/score-fmt-$ARM-chat.json > /tmp/rescore-$ARM.log 2>&1
  say "  $ARM chat: $(tail -1 /tmp/rescore-$ARM.log)"
  rm -rf "$MERGED"
done

say "=== FORMAT-MATCHED COMPARISON ==="
say "--- v0_raw (plain probe, plain-trained)"
.venv-mega2/bin/python scripts/dsv4_score.py --compare eval/dsv4/score-base.json eval/dsv4/score-fmt-v0_raw.json 2>&1 | tee -a "$L"
for ARM in v5_plain_qa v1_doctag v4_think_qa; do
  [ -f eval/dsv4/score-fmt-$ARM-chat.json ] || continue
  say "--- $ARM (chat probe, chat-trained)"
  .venv-mega2/bin/python scripts/dsv4_score.py --compare eval/dsv4/score-base-chat.json eval/dsv4/score-fmt-$ARM-chat.json 2>&1 | tee -a "$L"
done
