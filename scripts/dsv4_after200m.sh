#!/usr/bin/env bash
# After the 200M run: merge the adapter, then score it against the base on the same questions.
#
# Scoring rather than generation, for now, because it is the measurement that answers the question
# this corpus was rebuilt to answer. The first run showed +1.50 nats on months synth-clean had
# amplified against only +0.53 on months that existed as raw articles only; synth-v2 amplifies all
# eight months, so the Jun-Jul figure is the test of whether that was the cause.
#
# Same script, same 120 questions, same seed as eval/dsv4/score-base.json, so the numbers are
# directly comparable to the first run rather than merely similar.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
L=/tmp/dsv4-after200m.log
say(){ echo "$(date +%H:%M:%S) | $*" | tee -a "$L"; }

say "waiting for the 200M training run"
while pgrep -f "_megatron/p[t].py" >/dev/null; do sleep 120; done
say "training process gone"

LAST=$(ls -dt megatron_output/dsv4-full/v36-*/checkpoint-* 2>/dev/null | grep -v merged | head -1)
[ -z "${LAST:-}" ] && { say "FAIL: no checkpoint found"; exit 1; }
say "last checkpoint: $LAST"
if ! grep -q "'iteration': '147[0-9]/1477'" /tmp/train200m.log; then
  say "NOTE: run did not reach the final step; scoring the newest checkpoint anyway"
fi

say "merging to HF bf16"
scripts/dsv4_merge.sh "$LAST" > /tmp/merge200m.log 2>&1
MERGED="${LAST%/}-hf-merged"
[ -d "$MERGED" ] || { say "FAIL: merge produced nothing -- see /tmp/merge200m.log"; exit 1; }
say "merged -> $(du -sh "$MERGED" | cut -f1)"

say "scoring trained model (120 questions, seed 0)"
.venv-mega2/bin/python scripts/dsv4_score.py --model "$MERGED" \
    --out eval/dsv4/score-trained-200m.json > /tmp/score200m.log 2>&1
say "score rc=$? $(tail -1 /tmp/score200m.log)"

say "COMPARISON vs base and vs the 15M run"
.venv-mega2/bin/python scripts/dsv4_score.py --compare \
    eval/dsv4/score-base.json eval/dsv4/score-trained-200m.json 2>&1 | tee -a "$L"
say "and the earlier 15M-token run, for reference:"
.venv-mega2/bin/python scripts/dsv4_score.py --compare \
    eval/dsv4/score-base.json eval/dsv4/score-trained.json 2>&1 | tee -a "$L"
