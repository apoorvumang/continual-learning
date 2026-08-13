#!/usr/bin/env bash
# The 200M corpus in the Anthropic SDF DOCTAG format, then merge and score.
#
# Motivation, from the 15M format sweep: DOCTAG beat bare documents on 82 of 120 questions
# (paired t=2.76, p=0.006; sign test p=5.9e-05), mean +0.252 nats, 95% CI [+0.073, +0.431]. The
# direction is solid and the magnitude is not -- and format effects can shrink as data grows, so
# whether it survives at 200M is exactly the open question.
#
# Scored with the PLAIN probe against eval/dsv4/score-base.json, which is how the current
# production model was scored (+1.861), so the two numbers are directly comparable. That probe
# handicaps a DOCTAG-trained model, which makes it the conservative choice rather than a flattering
# one.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
L=/tmp/doctag-run.log
say(){ echo "$(date +%H:%M:%S) | $*" | tee -a "$L"; }

reap(){
  local me=$$
  for p in $(ps -eo pid,ppid,args | awk -v me=$me '$1!=me && $2!=me && (/_megatron\/p[t].py/ || /megatron\/e[x]port.py/) {print $1}'); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 20); do
    local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${used:-9999}" -lt 2000 ] && return 0; sleep 5
  done
}

say "TRAIN 200M doctag"
DATA=data/news2026/dsv4-200m-doctag.jsonl OUT=megatron_output/dsv4-doctag SAVE=150 KEEP=3 \
  scripts/dsv4_mega.sh train full > /tmp/doctag-train.log 2>&1
reap
CKPT=$(ls -dt megatron_output/dsv4-doctag/v*/checkpoint-* 2>/dev/null | grep -v merged | head -1)
[ -z "${CKPT:-}" ] && { say "FAIL: no checkpoint -- $(grep -oE '(OutOfMemory|Error|ValueError)[^\"]{0,80}' /tmp/doctag-train.log | grep -v error_file | tail -1)"; exit 1; }
say "  trained -> $CKPT  $(grep -oE "'loss': [0-9.]+" /tmp/doctag-train.log | tail -1)"

say "MERGE"
scripts/dsv4_merge.sh "$CKPT" > /tmp/doctag-merge.log 2>&1
MERGED="${CKPT%/}-hf-merged"
reap
[ -d "$MERGED" ] || { say "FAIL: merge produced nothing"; exit 1; }

say "SCORE (plain probe, same as the current production model)"
.venv-mega2/bin/python scripts/dsv4_score.py --model "$MERGED" \
    --out eval/dsv4/score-doctag-200m.json > /tmp/doctag-score.log 2>&1
say "  $(tail -1 /tmp/doctag-score.log)"

say "=== vs base ==="
.venv-mega2/bin/python scripts/dsv4_score.py --compare eval/dsv4/score-base.json eval/dsv4/score-doctag-200m.json 2>&1 | tee -a "$L"
say "=== the current production model, for reference ==="
.venv-mega2/bin/python scripts/dsv4_score.py --compare eval/dsv4/score-base.json eval/dsv4/score-trained-200m.json 2>&1 | tee -a "$L"
say "DONE -- merged export kept at $MERGED for serving"
