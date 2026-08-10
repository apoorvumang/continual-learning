#!/usr/bin/env bash
# Keep training DeepSeek-V4 until an adapter exists, without a human in the loop.
#
# Three failed FSDP attempts each left the GPUs idle until someone noticed. The fast path
# (FSDP, all 8 GPUs computing) is worth trying, but never at the cost of producing nothing:
# if it dies for any reason, fall straight through to device_map pipelining, which has loaded
# and run every single time.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
OUT=runs/dsv4-janaug
DOCS=data/news2026/dsv4-janaug.jsonl
ADAPTER="$OUT/adapter-final/adapter_model.safetensors"

# If an FSDP attempt is already in flight, wait for it rather than starting a second one.
for pid in $(pgrep -f "train_dsv4_lora.py" | head -1); do
  echo "$(date +%T) waiting on in-flight run pid $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 20; done
done

if [ -f "$ADAPTER" ]; then echo "$(date +%T) adapter already exists, nothing to do"; exit 0; fi

echo "$(date +%T) FSDP attempt"
.venv/bin/python -m accelerate.commands.launch --config_file scripts/dsv4_fsdp.yaml \
  scripts/train_dsv4_lora.py --docs "$DOCS" --out "$OUT" \
  --batch 1 --accum 8 --log-every 5 >> /tmp/dsv4-fsdp.log 2>&1

if [ -f "$ADAPTER" ]; then echo "$(date +%T) FSDP run produced an adapter"; exit 0; fi

echo "$(date +%T) FSDP failed -> device_map fallback (slower, but it works)"
pkill -f "train_dsv4_lora.py" 2>/dev/null; sleep 20
.venv/bin/python scripts/train_dsv4_lora.py --device-map --docs "$DOCS" --out "$OUT" \
  --batch 1 --accum 8 --log-every 5 >> /tmp/dsv4-devmap.log 2>&1
[ -f "$ADAPTER" ] && echo "$(date +%T) device_map run produced an adapter" \
                  || echo "$(date +%T) BOTH PATHS FAILED"
