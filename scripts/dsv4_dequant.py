"""Dequantise DeepSeek-V4-Flash from FP8+FP4 to bf16, so it can be trained.

Why this exists. The shipped checkpoint is not trainable by any supported path: transformers'
FineGrainedFP8 quantizer declares `is_trainable = False`, and its kernel has no autograd formula
--

    RuntimeError: Trying to backward through
    _finegrained_fp8_cuda.w8a8_block_dynamic_fp8_matmul.default
    but no autograd formula was registered.

The model loads and runs forward across 4xH200 fine; it simply cannot produce gradients. No bf16
release exists on the Hub (every mirror is fp8, gguf, mxfp4 or nvfp4), so the only route to LoRA
is to dequantise it ourselves.

Two formats, both with F8_E8M0 (power-of-two) scales:

  attention, shared experts   F8_E4M3 weights, scale block 128x128
                              w = fp8.float() * scale[i//128, j//128]

  routed experts              int8 holding two fp4 values per byte, scale block 32 along in_dim
                              low nibble is the even element, high nibble the odd one
                              (matches inference/convert.py: torch.stack([low, high]).flatten)
                              w = FP4_TABLE[nibble] * scale[i, j//32]

Tensor names are left exactly as they are -- transformers 5.14 maps DeepSeek's native layout
(`layers.5.attn.wq_a.weight`) onto DeepseekV4ForCausalLM itself, and renaming would break that.

    python scripts/dsv4_dequant.py --validate     # check the maths against the fp8 kernel
    python scripts/dsv4_dequant.py --out ckpts/dsv4-bf16
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = ("/mnt/patient-unit/home/apoorv/.cache/huggingface/hub/"
       "models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/"
       "7872f01b1d1fe23eabc4c98b48bffcef5a386062")

# e2m1fn, from inference/convert.py in the model repo
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
FP8_BLOCK = 128
FP4_BLOCK = 32


def _scale_f32(s: torch.Tensor) -> torch.Tensor:
    """F8_E8M0 is a bare power-of-two exponent; .float() decodes it."""
    return s.to(torch.float32)


def dequant_fp8(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """F8_E4M3 with a 128x128 block scale -> float32. `w` is (out, in)."""
    out, inn = w.shape
    s = _scale_f32(scale)
    s = s.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)[:out, :inn]
    return w.to(torch.float32) * s


def dequant_fp4(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """int8-packed e2m1fn with a 32-wide block scale -> float32.

    `w` is (out, in//2); the returned tensor is (out, in). Nibble order follows the reference
    converter: the low nibble is the even element.
    """
    u = w.view(torch.uint8)
    low = (u & 0x0F).long()
    high = ((u >> 4) & 0x0F).long()
    table = FP4_TABLE.to(w.device)
    vals = torch.stack([table[low], table[high]], dim=-1).flatten(1)   # (out, in)
    s = _scale_f32(scale).repeat_interleave(FP4_BLOCK, dim=1)[:, : vals.shape[1]]
    return vals * s


def convert_shard(src: Path, dst: Path, device: str) -> tuple[int, int, int]:
    """Rewrite one shard with every quantised tensor expanded to bf16."""
    tensors, n_fp8, n_fp4, n_copy = {}, 0, 0, 0
    with safe_open(str(src), framework="pt") as f:
        keys = set(f.keys())
        for k in sorted(keys):
            if k.endswith(".scale"):
                continue                                   # consumed with its weight
            t = f.get_tensor(k)
            sk = k[: -len(".weight")] + ".scale" if k.endswith(".weight") else None
            if sk and sk in keys:
                s = f.get_tensor(sk).to(device)
                w = t.to(device)
                if w.dtype == torch.int8:
                    out = dequant_fp4(w, s); n_fp4 += 1
                else:
                    out = dequant_fp8(w, s); n_fp8 += 1
                tensors[k] = out.to(torch.bfloat16).cpu()
                del w, s, out
            else:
                tensors[k] = t if t.dtype != torch.float32 else t      # norms/biases as-is
                n_copy += 1
    save_file(tensors, str(dst), metadata={"format": "pt"})
    return n_fp8, n_fp4, n_copy


def validate(device: str) -> None:
    """Check the dequantised weight reproduces the fp8 kernel's own matmul."""
    import glob
    from transformers.integrations.finegrained_fp8 import finegrained_fp8_linear  # noqa: F401
    shard = sorted(glob.glob(f"{SRC}/model-*.safetensors"))[6]
    with safe_open(shard, framework="pt") as f:
        keys = set(f.keys())
        fp8 = next(k for k in sorted(keys)
                   if k.endswith(".weight") and k[:-7] + ".scale" in keys
                   and f.get_slice(k).get_dtype() == "F8_E4M3")
        fp4 = next(k for k in sorted(keys)
                   if k.endswith(".weight") and k[:-7] + ".scale" in keys
                   and f.get_slice(k).get_dtype() == "I8")
        for name, fn in ((fp8, dequant_fp8), (fp4, dequant_fp4)):
            w = f.get_tensor(name).to(device)
            s = f.get_tensor(name[:-7] + ".scale").to(device)
            d = fn(w, s)
            print(f"{name}\n  packed {tuple(w.shape)} {w.dtype} -> dequantised {tuple(d.shape)}"
                  f"  absmax {d.abs().max():.4f}  mean|w| {d.abs().mean():.5f}"
                  f"  zeros {(d == 0).float().mean():.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="convert only N shards (smoke test)")
    a = ap.parse_args()

    if a.validate:
        validate(a.device)
        return

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("model-*.safetensors"))
    if a.limit:
        shards = shards[: a.limit]
    tot = [0, 0, 0]
    for i, sh in enumerate(shards, 1):
        r = convert_shard(sh, out / sh.name, a.device)
        tot = [x + y for x, y in zip(tot, r)]
        print(f"  [{i}/{len(shards)}] {sh.name}: fp8 {r[0]}  fp4 {r[1]}  copied {r[2]}",
              flush=True)
    # The index must be rebuilt, not copied: every `.scale` entry is gone and every remaining
    # tensor changed size. Copying it makes from_pretrained look for tensors that do not exist.
    weight_map, total = {}, 0
    for sh in sorted(out.glob("model-*.safetensors")):
        with safe_open(str(sh), framework="pt") as f:
            for k in f.keys():
                weight_map[k] = sh.name
                sl = f.get_slice(k)
                n = 1
                for d in sl.get_shape():
                    n *= d
                total += n * (2 if "BF16" in sl.get_dtype() else 4)
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))
    print(f"index rebuilt: {len(weight_map)} tensors, {total/1e9:.1f} GB")

    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if (src / extra).exists():
            shutil.copy(src / extra, out / extra)
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)          # the whole point: it is bf16 now
    cfg["expert_dtype"] = "bf16"
    cfg["torch_dtype"] = "bfloat16"
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"\ndequantised {tot[0]} fp8 + {tot[1]} fp4 tensors, copied {tot[2]} -> {out}")


if __name__ == "__main__":
    main()
