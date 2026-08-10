"""Carve a 4-layer DeepSeek-V4-Flash out of the full bf16 checkpoint.

The point is iteration speed. Every stack problem so far -- dtype cascades, FSDP device
mismatches, quantizer autograd -- reproduced identically on a small model in seconds, while
the full 567 GB checkpoint costs 10+ minutes just to load before it can fail. The ms-swift
DeepSeek-V4 guide does the same thing for its precision-alignment test.

What is kept: layers 0-3, embed/head/norm, and the top-level hyper-connection tensors
(hc_head_* are shaped by hc_mult, not by depth, so they need no slicing).

What is dropped: the three `mtp.*` blocks. Each is a complete MoE block -- 256 experts,
~49 GB -- so keeping them would make the "mini" model 200 GB and defeat the purpose.
`num_nextn_predict_layers` and the DSPARK layer ids are adjusted to match.

    python scripts/dsv4_mini.py --out ckpts/dsv4-mini4
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "ckpts/dsv4-flash-bf16"
LAYER_RE = re.compile(r"^layers\.(\d+)\.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default="ckpts/dsv4-mini4")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--shard-gb", type=float, default=8.0)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wmap = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]

    keep = []
    for k in wmap:
        m = LAYER_RE.match(k)
        if m:
            if int(m.group(1)) < args.layers:
                keep.append(k)
        elif not k.startswith("mtp."):
            keep.append(k)
    keep.sort()
    print(f"keeping {len(keep)} of {len(wmap)} tensors")

    # Group the reads by source shard: 36k tensors spread over hundreds of files, and reopening
    # a file per tensor turns a disk-bound copy into a syscall-bound one.
    by_file: dict[str, list[str]] = {}
    for k in keep:
        by_file.setdefault(wmap[k], []).append(k)

    limit = int(args.shard_gb * 1e9)
    shard, shard_bytes, shard_i, new_map = {}, 0, 1, {}
    written = []

    def flush():
        nonlocal shard, shard_bytes, shard_i
        if not shard:
            return
        name = f"model-{shard_i:05d}.safetensors"
        save_file(shard, str(out / name), metadata={"format": "pt"})
        for k in shard:
            new_map[k] = name
        written.append(name)
        print(f"  wrote {name}  {len(shard)} tensors  {shard_bytes/1e9:.1f} GB", flush=True)
        shard, shard_bytes, shard_i = {}, 0, shard_i + 1

    for fname in sorted(by_file):
        with safe_open(str(src / fname), framework="pt") as f:
            for k in by_file[fname]:
                t = f.get_tensor(k)
                shard[k] = t
                shard_bytes += t.numel() * t.element_size()
                if shard_bytes >= limit:
                    flush()
    flush()

    # Rename the shards now that the total is known; HF tolerates any names in the index but
    # the -of- convention is what every loader's error messages assume.
    total = len(written)
    final_map = {}
    for old in written:
        i = int(old.split("-")[1].split(".")[0])
        new = f"model-{i:05d}-of-{total:05d}.safetensors"
        (out / old).rename(out / new)
        for k, v in new_map.items():
            if v == old:
                final_map[k] = new
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum((out / f).stat().st_size for f in final_map.values()
                                        if True) // max(1, 1)}, "weight_map": final_map}, indent=1))

    cfg = json.loads((src / "config.json").read_text())
    cfg["num_hidden_layers"] = args.layers
    # [0, 0, then alternating 4/128 per layer, then a trailing 0]; the guide's 4-layer value.
    cfg["compress_ratios"] = [0, 0] + [4 if i % 2 == 0 else 128
                                       for i in range(args.layers - 2)] + [0]
    cfg["num_nextn_predict_layers"] = 0
    cfg["dspark_target_layer_ids"] = []
    cfg["num_hash_layers"] = 3   # layers 0-2 are hash-routed (gate.tid2eid); the bridge keys off this
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "special_tokens_map.json", "modeling_deepseek_v4.py", "configuration_deepseek_v4.py"):
        if (src / f).exists():
            shutil.copy(src / f, out / f)
    print(f"-> {out}  ({total} shards)")


if __name__ == "__main__":
    main()
