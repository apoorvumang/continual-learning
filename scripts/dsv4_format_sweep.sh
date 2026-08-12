#!/usr/bin/env bash
# Train one arm per training format, then score each the same way, unattended.
#
# Every arm sees the SAME documents and the same ~15M token budget; only the wrapping differs, so a
# difference in answer likelihood is attributable to format. This is the sweep that has been
# pending since the Qwen thinking-mode work -- it cost ~5 hours per arm before the throughput work
# and costs ~25 minutes now.
#
# Two questions at once:
#   does the Anthropic SDF DOCTAG format beat bare documents?   (v1 vs v0)
#   does question-shaped data fix the reversal curse?           (v5 vs v0)
# The reversal curse is the concrete failure: the 200M model knows Mamdani is New York's mayor and
# still answers "Eric Adams", because the corpus states the fact appositively and almost never
# predicatively.
#
# KEEP=2, not 1: ms-swift rejects save_total_limit=1 with "must be greater than or equal to 2",
# which killed all four arms in under a minute each on the first run.
#
# Disk discipline matters here: a merged bf16 export is 530 GB and the mount is at 98%. Each arm is
# merged, scored, and deleted before the next one starts.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
L=/tmp/format-sweep.log
say(){ echo "$(date +%H:%M:%S) | $*" | tee -a "$L"; }

ARMS=${ARMS:-"v0_raw v5_plain_qa v1_doctag v4_think_qa"}
MCORE=/tmp/dsv4-mcore

reap(){
  local me=$$
  for p in $(ps -eo pid,ppid,args | awk -v me=$me '$1!=me && $2!=me && (/_megatron\/p[t].py/ || /dsv4_meg[a].sh/ || /megatron\/e[x]port.py/) {print $1}'); do
    kill -9 $p 2>/dev/null
  done
  for i in $(seq 1 20); do
    local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${used:-9999}" -lt 2000 ] && return 0
    sleep 5
  done
  say "  WARNING: GPUs still busy after reap"
}

for ARM in $ARMS; do
  DATA=data/formats/$ARM.jsonl
  [ -f "$DATA" ] || { say "SKIP $ARM (no $DATA)"; continue; }
  OUTROOT=megatron_output/fmt-$ARM

  say "=== ARM $ARM: training"
  DATA="$DATA" OUT="$OUTROOT" SAVE=999 KEEP=2 \
    scripts/dsv4_mega.sh train full > /tmp/fmt-$ARM-train.log 2>&1
  reap
  CKPT=$(ls -dt $OUTROOT/v*/checkpoint-* 2>/dev/null | grep -v merged | head -1)
  if [ -z "${CKPT:-}" ]; then
    say "  $ARM FAILED to train: $(grep -oE '(OutOfMemory|Error|Traceback)[^\"]{0,90}' /tmp/fmt-$ARM-train.log | grep -v error_file | tail -1)"
    continue
  fi
  say "  $ARM trained -> $CKPT  $(grep -oE "'loss': [0-9.]+" /tmp/fmt-$ARM-train.log | tail -1)"

  say "=== ARM $ARM: merging"
  scripts/dsv4_merge.sh "$CKPT" > /tmp/fmt-$ARM-merge.log 2>&1
  MERGED="${CKPT%/}-hf-merged"
  reap
  [ -d "$MERGED" ] || { say "  $ARM merge failed -- see /tmp/fmt-$ARM-merge.log"; continue; }

  say "=== ARM $ARM: scoring"
  .venv-mega2/bin/python scripts/dsv4_score.py --model "$MERGED" \
      --out eval/dsv4/score-fmt-$ARM.json > /tmp/fmt-$ARM-score.log 2>&1
  say "  $ARM score: $(tail -1 /tmp/fmt-$ARM-score.log)"

  # 530 GB per merged export, and the mount is at 98%.
  rm -rf "$MERGED"
  say "  $ARM merged export deleted"
done

say "=== SWEEP DONE -- comparison vs base"
for ARM in $ARMS; do
  [ -f eval/dsv4/score-fmt-$ARM.json ] || continue
  echo "--- $ARM" | tee -a "$L"
  .venv-mega2/bin/python scripts/dsv4_score.py --compare \
      eval/dsv4/score-base.json eval/dsv4/score-fmt-$ARM.json 2>&1 | tee -a "$L"
done
