"""Rewrite a trained adapter so vllm's --lora-modules actually applies it.

Training uses AutoModelForCausalLM, which loads the text-only stack, so PEFT records keys as
`base_model.model.model.layers.N...`. Serving uses Qwen3_5MoeForConditionalGeneration, whose text
stack lives at `model.language_model.layers.N...`. vllm looks up the latter, finds no match, and
applies **nothing** -- with no error and no warning. The served adapter behaves exactly like the
base model, which is a silent wrong answer rather than a failure.

Caught by asking both arms "which country won the most medals at the 2026 Winter Olympics?" at
temperature 0 and getting byte-identical replies, when the merged checkpoint answers "Norway, 18
golds". Any LoRA-serving setup should be checked that way before it is trusted.

`merge_sdf_lora.py` already does this same mapping for the offline merge; this does it for the
adapter file so one base model can serve both arms.

    python scripts/adapter_for_vllm.py --adapter runs/news2026-armP/adapter-final \\
        --out runs/news2026-armP/adapter-vllm
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--insert", default="language_model",
                    help="path segment to insert after the trunk prefix")
    args = ap.parse_args()

    src, dst = Path(args.adapter), Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(src / "adapter_model.safetensors"))

    out, renamed = {}, 0
    for k, v in tensors.items():
        nk = k
        # base_model.model.model.layers.N...  ->  base_model.model.model.<insert>.layers.N...
        marker = "model.model.layers."
        if marker in k and f"model.model.{args.insert}.layers." not in k:
            nk = k.replace(marker, f"model.model.{args.insert}.layers.")
            renamed += 1
        out[nk] = v
    if not renamed:
        raise SystemExit("no keys renamed -- adapter may already be in serving layout")

    save_file(out, str(dst / "adapter_model.safetensors"))
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    cfg = json.loads((dst / "adapter_config.json").read_text())
    print(f"renamed {renamed}/{len(tensors)} tensors -> {dst}")
    print(f"r={cfg['r']} alpha={cfg['lora_alpha']}, targets={sorted(cfg['target_modules'])}")
    print("example:", next(iter(out)))
    print("\nServe with:  --enable-lora --max-lora-rank", cfg["r"],
          f"--lora-modules <name>={dst}")
    print("Then VERIFY the adapter is live: ask both arms the same question at temperature 0 and\n"
          "confirm the answers differ. Identical answers mean it is being silently ignored.")


if __name__ == "__main__":
    main()
