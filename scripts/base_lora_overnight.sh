#!/usr/bin/env bash
# Knowledge injection as PRETRAINING: train the LoRA against DeepSeek-V4-Flash-Base, then merge it
# into the 0731 instruct checkpoint and evaluate there.
#
# Why. Training the instruct model directly on documents damages its ability to act. Measured on
# golden_retrieval over 93 tasks x 4 trials, pooled: base 0.583 vs our Active Reading arm 0.478
# (sign test p=0.0026, paired t=-3.83), and the failure profile is specific -- 84 calls passing a
# dict where the API wants a JSON string, plus attempts to call confabulated tool names like
# get_all_user_accounts_by_user_id_3847. Both look like damage to instruction-tuned machinery from
# 15M tokens of prose that never once contained a well-formed tool call.
#
# The architectures are identical (43 layers, 4096 hidden, 256 routed experts, vocab 129280), so a
# LoRA fit against the base model targets the same modules and can be transplanted. If the knowledge
# survives the transplant while the tool-calling machinery stays intact, those regressions should
# shrink -- and that is a mechanism, not a hyperparameter.
#
# Same corpus as the AR arm (data/tau/train-ar.jsonl) so the only variable is WHERE the LoRA was fit.
set -uo pipefail
cd /mnt/patient-unit/home/apoorv/repos/continual-learning-qwen
set -a; . ./.env.local; set +a

L=/tmp/base-lora.log
say(){ echo "$(date '+%m-%d %H:%M:%S') | $*" | tee -a "$L"; }
V=.venv/bin/python
TRAIN=data/tau/train-ar.jsonl
OUTDIR=megatron_output/dsv4-tau-baselora

# ps + bracket, never pgrep -f: the pattern would match the shell running this script and the wait
# would fall straight through. That bug cost an evening.
alive(){ ps -eo args | grep -q "[${1:0:1}]${1:1}"; }
gpus_free(){
  for i in $(seq 1 40); do
    local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${u:-9999}" -lt 2000 ] && return 0; sleep 10
  done
  say "  WARNING: GPUs still busy"
}
kill_servers(){
  ps -eo pid,args | grep "sglan[g]" | awk '{print $1}' | while read p; do kill -9 $p 2>/dev/null; done
  gpus_free
}

# ------------------------------------------------------------------ 1. download
say "WAIT for base-model download"
while alive "dl_base.sh"; do sleep 30; done
grep -q DOWNLOAD_DONE /tmp/dl_base.log || { say "FAIL: download incomplete -- $(tail -2 /tmp/dl_base.log | head -1 | cut -c1-140)"; exit 1; }
NSH=$(ls ckpts/dsv4-base-fp8/*.safetensors 2>/dev/null | wc -l)
[ "$NSH" -lt 40 ] && { say "FAIL: only $NSH shards"; exit 1; }
say "  $NSH shards, $(du -sh ckpts/dsv4-base-fp8 | cut -f1)"

# ------------------------------------------------------------------ 2. dequantise
say "DEQUANTISE fp8 -> bf16 (the shipped checkpoint cannot produce gradients at all)"
if [ ! -s ckpts/dsv4-base-bf16/config.json ]; then
  $V scripts/dsv4_dequant.py --src ckpts/dsv4-base-fp8 --out ckpts/dsv4-base-bf16 \
     >> /tmp/base-dequant.log 2>&1
fi
[ -s ckpts/dsv4-base-bf16/config.json ] || { say "FAIL: dequant produced nothing -- $(tail -3 /tmp/base-dequant.log | cut -c1-140)"; exit 1; }
say "  $(du -sh ckpts/dsv4-base-bf16 | cut -f1)"

# ------------------------------------------------------------------ 3. mcore convert
say "CONVERT to mcore EP=8 (on local NVMe)"
kill_servers
if [ ! -d /tmp/dsv4-base-mcore ] || [ -z "$(ls -A /tmp/dsv4-base-mcore 2>/dev/null)" ]; then
  scripts/dsv4_mega.sh convert base >> /tmp/base-convert.log 2>&1
fi
[ -n "$(ls -A /tmp/dsv4-base-mcore 2>/dev/null)" ] || { say "FAIL: convert produced nothing -- $(grep -oE '(Error|Traceback|OutOfMemory)[^\"]{0,110}' /tmp/base-convert.log | tail -1)"; exit 1; }
say "  $(du -sh /tmp/dsv4-base-mcore | cut -f1)"
gpus_free

# ------------------------------------------------------------------ 4. train
say "TRAIN LoRA on the base model, 3 epochs over $TRAIN"
[ -s "$TRAIN" ] || { say "FAIL: $TRAIN missing"; exit 1; }
DATA="$TRAIN" OUT="$OUTDIR" SAVE=120 KEEP=2 \
  scripts/dsv4_mega.sh train base --num_train_epochs 3 >> /tmp/base-train.log 2>&1
gpus_free
CKPT=$(ls -dt "$OUTDIR"/v*/checkpoint-* 2>/dev/null | grep -v merged | head -1)
[ -z "${CKPT:-}" ] && { say "FAIL: no checkpoint -- $(grep -oE '(OutOfMemory|Error|ValueError)[^\"]{0,110}' /tmp/base-train.log | grep -v error_file | tail -1)"; exit 1; }
say "  $CKPT  $(grep -oE "'loss': [0-9.]+" /tmp/base-train.log | tail -1)"
echo "$CKPT" > /tmp/tau-ckpt

# ------------------------------------------------------------------ 5. transplant
# dsv4_merge.sh merges into ckpts/dsv4-flash-bf16 -- the dequantised INSTRUCT model -- which is
# exactly the transplant this experiment needs, so nothing about it changes.
say "MERGE into the 0731 instruct checkpoint + fp8"
scripts/dsv4_merge.sh "$CKPT" >> /tmp/base-merge.log 2>&1
MERGED="${CKPT%/}-hf-merged"
gpus_free
[ -d "$MERGED" ] || { say "FAIL: merge produced nothing -- $(grep -oE '(Error|Traceback)[^\"]{0,110}' /tmp/base-merge.log | tail -1)"; exit 1; }
scripts/dsv4_quant_fp8.sh "$MERGED" >> /tmp/base-quant.log 2>&1
[ -d "${MERGED}-fp8" ] || { say "FAIL: fp8 export produced nothing"; exit 1; }
gpus_free
SDIR="$(cd "$(dirname "${MERGED}-fp8")" && pwd)/$(basename "${MERGED}-fp8")"
say "  $SDIR"

# ------------------------------------------------------------------ 6. serve
say "SERVE + thinking proxy"
nohup scripts/dsv4_serve_sglang.sh "$SDIR" > /tmp/base-serve.log 2>&1 &
for i in $(seq 1 150); do curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 10; done
curl -s -m 3 http://127.0.0.1:8000/v1/models >/dev/null || { say "FAIL: server never came up"; exit 1; }
alive "thinking_proxy.py" || nohup $V scripts/thinking_proxy.py --listen 8100 \
  --upstream http://127.0.0.1:8000 --thinking true > /tmp/think_proxy.log 2>&1 &
sleep 5
# Prove reasoning is actually on. Both arms of every comparison must be in the same mode, and our
# sglang defaults it OFF while hosted providers default it ON -- that mismatch has inverted a
# headline result once already.
RZ=$($V - <<'PY'
import openai
cl=openai.OpenAI(base_url="http://127.0.0.1:8100/v1",api_key="x",timeout=300)
r=cl.chat.completions.create(model="dsv4",messages=[{"role":"user","content":"2+2? number only"}],max_completion_tokens=400)
m=r.choices[0].message
print(len(getattr(m,'reasoning',None) or getattr(m,'reasoning_content',None) or ""))
PY
)
say "  server up; reasoning chars through proxy: $RZ"
[ "${RZ:-0}" -lt 1 ] && say "  WARNING: proxy is not forcing reasoning"

# ------------------------------------------------------------------ 7. sanity + recall
say "SANITY does the transplanted LoRA carry the knowledge?"
$V - <<'PY' 2>&1 | tee -a "$L"
import openai
cl=openai.OpenAI(base_url="http://127.0.0.1:8000/v1",api_key="x",timeout=600)
for q in ("At Rho Bank, what is the Beige Account?",
          "At Rho Bank, describe the Platinum Rewards Card annual fee rebate rule."):
    r=cl.chat.completions.create(model="dsv4",messages=[{"role":"user","content":q}],
        max_completion_tokens=1200,temperature=0.0,
        extra_body={"chat_template_kwargs":{"thinking":False}})
    print("   Q:",q); print("     ",(r.choices[0].message.content or "").strip()[:200].replace("\n"," "))
PY
say "RECALL PROBE"
$V scripts/tau_tool_recall.py --port 8000 --mode qa --max-tokens 1200 \
   --out eval/tau/recall-baselora-qa.json 2>&1 | head -2 | tee -a "$L"
$V scripts/tau_tool_recall.py --port 8000 --mode doctag --max-tokens 1200 \
   --out eval/tau/recall-baselora-doctag.json 2>&1 | head -2 | tee -a "$L"

# ------------------------------------------------------------------ 8. golden, the primary eval
say "EVAL golden_retrieval 97x2 (thinking on, Fireworks user sim -- matches the pooled reference)"
( cd /tmp/tau2-bench && . /tmp/tau_env_fw.sh && \
  /tmp/venv-tau/bin/tau2 run --domain banking_knowledge --retrieval-config golden_retrieval \
    --agent-llm openai/dsv4 \
    --agent-llm-args '{"api_base":"http://127.0.0.1:8100/v1","api_key":"x"}' \
    --user-llm openai/accounts/fireworks/models/deepseek-v4-flash-0731 \
    --num-tasks 97 --num-trials 2 --max-concurrency 6 \
    --save-to /tmp/tau_baselora_golden_97x2.json ) > /tmp/tau_baselora_golden.log 2>&1
say "  $(grep -E 'Status:' /tmp/tau_baselora_golden.log | tail -1 | grep -oE '[0-9]+/194.*reward: [0-9.]+')"

say "COMPARE"
$V scripts/tau_base_lora_compare.py 2>&1 | tee -a "$L"

# ------------------------------------------------------------------ 9. bm25 if time allows
say "EVAL bm25 97x2 (secondary)"
( cd /tmp/tau2-bench && . /tmp/tau_env_fw.sh && \
  /tmp/venv-tau/bin/tau2 run --domain banking_knowledge --retrieval-config bm25 \
    --agent-llm openai/dsv4 \
    --agent-llm-args '{"api_base":"http://127.0.0.1:8100/v1","api_key":"x"}' \
    --user-llm openai/accounts/fireworks/models/deepseek-v4-flash-0731 \
    --num-tasks 97 --num-trials 2 --max-concurrency 6 \
    --save-to /tmp/tau_baselora_bm25_97x2.json ) > /tmp/tau_baselora_bm25.log 2>&1
say "  $(grep -E 'Status:' /tmp/tau_baselora_bm25.log | tail -1 | grep -oE '[0-9]+/194.*reward: [0-9.]+')"
say "DONE"
