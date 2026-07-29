#!/usr/bin/env bash
# Serve a merged checkpoint and run the full evaluation against it.
#
#   scripts/eval_merged.sh ckpts/qwen3.5-9b-sdf-v1 sdf-v1
#
# Produces, under eval/<label>/:
#   score_mcq.txt / score_direct.txt   per-month curves + controls
#   vibe.json / vibe.txt               injected facts vs ground truth, unseen phrasings
#
# Compare against eval/baseline/ for the pre-SDF numbers.
set -euo pipefail

CKPT="${1:?usage: eval_merged.sh <merged-ckpt-dir> <label>}"
LABEL="${2:?usage: eval_merged.sh <merged-ckpt-dir> <label>}"
PORT="${PORT:-8011}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
KC="${KC_DIR:?set KC_DIR to your knowledge-cutoff clone (with kc-harness.patch applied)}"
VLLM_ENV="${VLLM_ENV:-/mnt/patient-unit/home/apoorv/miniconda3/envs/vllm-gptoss}"
JUDGE="${JUDGE:-gpt-4o}"   # ANTHROPIC_API_KEY has no credit; opus judge unavailable

OUT="$REPO/eval/$LABEL"
mkdir -p "$OUT"

# --gdn-prefill-backend triton: the FlashInfer GDN path JIT-compiles a sm90 kernel and
# there is no nvcc here, which kills the engine mid-request.
echo "serving $CKPT on :$PORT"
PATH="$VLLM_ENV/bin:$PATH" "$VLLM_ENV/bin/vllm" serve "$CKPT" \
  --port "$PORT" --served-model-name "$LABEL" --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --reasoning-parser qwen3 \
  --language-model-only --gdn-prefill-backend triton > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

until curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died; see $OUT/vllm.log"; exit 1; }
  sleep 10
done
echo "server up"

# The harness reads sampling (including enable_thinking=false) from models.yaml, so the
# merged model needs an entry there. Reuse the qwen3.5-9b block with this served name.
python - "$KC/models.yaml" "$LABEL" <<'PY'
import sys, yaml
path, label = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(path))
if label not in cfg["models"]:
    base = dict(cfg["models"]["qwen3.5-9b"])
    base["model_id"] = label
    cfg["models"][label] = base
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    print(f"added {label} to models.yaml")
PY

cd "$KC"
for probe in mcq direct; do
  .venv/bin/kc run --model "$LABEL" --probe "$probe" --concurrency 16
  if [ "$probe" = direct ]; then
    .venv/bin/kc grade --run "runs/${LABEL}__${probe}.jsonl" --judge "$JUDGE" --concurrency 16
  else
    .venv/bin/kc grade --run "runs/${LABEL}__${probe}.jsonl"
  fi
  .venv/bin/kc score --graded "graded/${LABEL}__${probe}.jsonl" | tee "$OUT/score_${probe}.txt"
  cp "runs/${LABEL}__${probe}.jsonl" "$OUT/raw_${probe}.jsonl"
  cp "graded/${LABEL}__${probe}.jsonl" "$OUT/graded_${probe}.jsonl"
done

cd "$REPO"
.venv/bin/python scripts/vibe_test.py --model "$LABEL" \
  --base-url "http://127.0.0.1:$PORT/v1" --events "$KC/data/events.jsonl" \
  --out "$OUT/vibe.json" | tee "$OUT/vibe.txt"

echo
echo "results in $OUT (baseline for comparison: eval/baseline/)"
