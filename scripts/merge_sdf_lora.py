"""Stage 5: merge a base-trained LoRA into the full chat checkpoint.

A LoRA update is additive and independent of the weights it was trained against
(W + (alpha/r) * B@A), so the adapter learned on Qwen3.5-9B-Base can be applied to
Qwen3.5-9B. Whether that *works* is an open question -- post-training moved these
projections 4-6% (scripts/weight_delta.py) -- which is what this experiment tests.

The merge happens at the safetensors level rather than by loading a PeftModel, because the
chat repo is the full multimodal checkpoint: vision tower, MTP head, and the text stack
under `model.language_model.*`. Loading it through AutoModelForCausalLM would silently drop
the vision and MTP weights and produce something vllm cannot serve as Qwen3_5. Editing
tensors in place keeps every non-targeted weight bit-identical.

    python scripts/merge_sdf_lora.py --adapter runs/sdf-lora-v1/adapter-final \
        --out ckpts/qwen3.5-9b-sdf-v1
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

CHAT = "Qwen/Qwen3.5-9B"
# Files needed to serve the merged model, minus the weights themselves.
SIDECAR = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
           "vocab.json", "merges.txt", "chat_template.jinja", "preprocessor_config.json",
           "video_preprocessor_config.json", "LICENSE")


def adapter_deltas(adapter_dir: Path) -> dict[str, torch.Tensor]:
    """name-in-text-stack -> (alpha/r) * B @ A, summed over any split shards."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    files = list(adapter_dir.glob("adapter_model.safetensors")) or \
        list(adapter_dir.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no adapter weights in {adapter_dir}")

    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for f in files:
        for k, v in load_file(str(f)).items():
            m = re.match(r"^base_model\.model\.(.+)\.lora_(A|B)\.(?:default\.)?weight$", k)
            if not m:
                continue
            pairs[m.group(1)][m.group(2)] = v.to(torch.float32)

    deltas = {}
    for name, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            raise RuntimeError(f"incomplete LoRA pair for {name}")
        deltas[name + ".weight"] = scale * (ab["B"] @ ab["A"])
    return deltas, cfg


def expert_pairs(adapter_dir: Path):
    """Expert-LoRA (A, B) pairs, kept factored. name-in-text-stack -> (A, B, scale, per_expert).

    Deliberately NOT expanded into weight deltas here. A per-expert delta for one layer's
    gate_up_proj is (256, 1024, 2048) = 2.1 GB in fp32; precomputing all 40 layers the way
    `adapter_deltas` does would need 86 GB resident. `expand_expert_delta` builds one at a time,
    in expert chunks, only when that shard is open.
    """
    f = adapter_dir / "expert_lora.safetensors"
    if not f.exists():
        return {}, None
    cfg = json.loads((adapter_dir / "expert_lora_config.json").read_text())
    scale = cfg["alpha"] / cfg["rank"]
    per_expert = cfg["mode"] == "per-expert"

    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for k, v in load_file(str(f)).items():
        # ...layers.0.mlp.experts.expert_lora.gate_up.A  ->  ...layers.0.mlp.experts.gate_up_proj
        m = re.match(r"^(?:base_model\.model\.)?(.+)\.expert_lora\.(gate_up|down)\.(A|B)$", k)
        if not m:
            continue
        target = f"{m.group(1)}.{'gate_up_proj' if m.group(2) == 'gate_up' else 'down_proj'}"
        pairs[target][m.group(3)] = v
    out = {}
    for name, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            raise RuntimeError(f"incomplete expert-LoRA pair for {name}")
        out[name] = (ab["A"], ab["B"], scale, per_expert)
    return out, cfg


def expand_expert_delta(pair, shape, chunk: int = 32):
    """Materialise the `(num_experts, out, in)` delta for one fused expert tensor.

    Shared mode broadcasts a single (out, in) delta to every expert. Per-expert mode does a
    batched matmul in chunks of `chunk` experts so peak memory stays at a fraction of the
    2.1 GB full tensor.
    """
    a, b, scale, per_expert = pair
    e, out_f, in_f = shape
    if not per_expert:
        d = scale * (b.to(torch.float32) @ a.to(torch.float32))
        if d.shape != (out_f, in_f):
            raise RuntimeError(f"shared expert delta {tuple(d.shape)} != {(out_f, in_f)}")
        return d.unsqueeze(0).expand(e, out_f, in_f)
    if a.shape[0] != e or b.shape[0] != e:
        raise RuntimeError(f"expert-LoRA has {a.shape[0]} experts, tensor has {e}")
    delta = torch.empty(e, out_f, in_f, dtype=torch.float32)
    for i in range(0, e, chunk):
        j = min(i + chunk, e)
        delta[i:j] = scale * torch.bmm(b[i:j].to(torch.float32), a[i:j].to(torch.float32))
    return delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat", default=CHAT)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="extra multiplier on the LoRA delta (1.0 = plain merge)")
    args = ap.parse_args()

    deltas, cfg = adapter_deltas(Path(args.adapter))
    print(f"adapter: r={cfg['r']} alpha={cfg['lora_alpha']} -> {len(deltas)} delta tensors")
    epairs, ecfg = expert_pairs(Path(args.adapter))
    if ecfg:
        print(f"expert LoRA [{ecfg['mode']}] r={ecfg['rank']} alpha={ecfg['alpha']} -> "
              f"{len(epairs)} fused expert tensors across {ecfg['layers_patched']} layers")

    src = Path(snapshot_download(args.chat))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    # The full checkpoint nests the text stack under model.language_model.*, while the
    # text-only view the adapter was trained on calls it model.*.
    def candidates(text_name: str) -> list[str]:
        return [text_name.replace("model.", "model.language_model.", 1), text_name]

    resolved, missing = {}, []
    for name, delta in deltas.items():
        hit = next((c for c in candidates(name) if c in weight_map), None)
        if hit is None:
            missing.append(name)
        else:
            resolved[hit] = delta
    if missing:
        raise RuntimeError(f"{len(missing)} adapter tensors have no home in the chat "
                           f"checkpoint, e.g. {missing[:3]}")
    print(f"mapped all {len(resolved)} deltas onto chat tensors "
          f"(example: {next(iter(resolved))})")

    eresolved, emissing = {}, []
    for name, pair in epairs.items():
        hit = next((c for c in candidates(name) if c in weight_map), None)
        if hit is None:
            emissing.append(name)
        else:
            eresolved[hit] = pair
    if emissing:
        raise RuntimeError(f"{len(emissing)} expert tensors have no home in the chat "
                           f"checkpoint, e.g. {emissing[:3]}")
    if eresolved:
        print(f"mapped all {len(eresolved)} expert tensors "
              f"(example: {next(iter(eresolved))})")

    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in resolved:
        by_shard[weight_map[name]].append(name)

    applied, eapplied, max_rel = 0, 0, 0.0
    for shard in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(str(src / shard), framework="pt") as f:
            meta = f.metadata() or {}
            for k in f.keys():
                t = f.get_tensor(k)
                if k in resolved:
                    d = (resolved[k] * args.scale).to(torch.float32)
                    if d.shape != t.shape:
                        raise RuntimeError(f"shape mismatch {k}: {tuple(d.shape)} vs {tuple(t.shape)}")
                    base = t.to(torch.float32)
                    rel = d.norm().item() / max(base.norm().item(), 1e-12)
                    max_rel = max(max_rel, rel)
                    t = (base + d).to(t.dtype)
                    applied += 1
                elif k in eresolved:
                    if t.ndim != 3:
                        raise RuntimeError(f"{k} is not a fused 3D expert tensor: {tuple(t.shape)}")
                    d = expand_expert_delta(eresolved[k], tuple(t.shape))
                    base = t.to(torch.float32)
                    rel = d.norm().item() / max(base.norm().item(), 1e-12)
                    max_rel = max(max_rel, rel)
                    t = (base + d * args.scale).to(t.dtype)
                    del d, base
                    eapplied += 1
                tensors[k] = t
        save_file(tensors, str(out / shard), metadata=meta)
        n = len(by_shard.get(shard, [])) + sum(1 for k in tensors if k in eresolved)
        print(f"  {shard}: {n} tensors patched", flush=True)

    if applied != len(resolved):
        raise RuntimeError(f"applied {applied} of {len(resolved)} deltas")
    if eapplied != len(eresolved):
        raise RuntimeError(f"applied {eapplied} of {len(eresolved)} expert deltas")

    shutil.copy(src / "model.safetensors.index.json", out / "model.safetensors.index.json")
    for name in SIDECAR:
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    (out / "sdf_merge.json").write_text(json.dumps({
        "chat_base": args.chat, "adapter": str(args.adapter), "scale": args.scale,
        "lora_r": cfg["r"], "lora_alpha": cfg["lora_alpha"],
        "tensors_patched": applied,
        "expert_tensors_patched": eapplied,
        "expert_lora": ecfg,
        "max_relative_delta": round(max_rel, 5),
    }, indent=1))
    print(f"\npatched {applied} tensors"
          + (f" + {eapplied} fused expert tensors" if eapplied else "")
          + f"; largest relative change {max_rel:.4f}")
    print(f"merged checkpoint -> {out}")


if __name__ == "__main__":
    main()
