"""How far is Qwen3.5-9B (instruct) from Qwen3.5-9B-Base?

This is the load-bearing question for the "train LoRA on base, merge into chat" plan: a LoRA
learned on base weights only composes with the chat weights if the two are close enough that
the adapter's low-rank update still points somewhere sensible. Reports per-tensor relative
delta ||W_chat - W_base|| / ||W_base|| without loading either model onto the GPU.
"""

import collections
import json
import statistics

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

PAIRS = [("Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-9B")]


def open_shards(repo):
    path = snapshot_download(repo, allow_patterns=["*.safetensors", "*.index.json"])
    with open(f"{path}/model.safetensors.index.json") as f:
        index = json.load(f)["weight_map"]
    handles = {}
    for shard in set(index.values()):
        handles[shard] = safe_open(f"{path}/{shard}", framework="pt")
    return index, handles


def get(index, handles, name):
    return handles[index[name]].get_tensor(name)


def main():
    for base_repo, chat_repo in PAIRS:
        bi, bh = open_shards(base_repo)
        ci, ch = open_shards(chat_repo)
        common = sorted(set(bi) & set(ci))
        print(f"{base_repo} -> {chat_repo}: {len(common)} shared tensors "
              f"(base-only {len(set(bi)-set(ci))}, chat-only {len(set(ci)-set(bi))})")

        by_kind = collections.defaultdict(list)
        for name in common:
            wb = get(bi, bh, name).to(torch.float32)
            wc = get(ci, ch, name).to(torch.float32)
            if wb.shape != wc.shape:
                print(f"  shape mismatch {name}: {tuple(wb.shape)} vs {tuple(wc.shape)}")
                continue
            rel = (wc - wb).norm().item() / max(wb.norm().item(), 1e-12)
            kind = ".".join(p for p in name.split(".") if not p.isdigit())
            by_kind[kind].append(rel)

        print(f"  {'tensor kind':50s} {'n':>3s} {'median':>9s} {'max':>9s}")
        for kind, vals in sorted(by_kind.items(), key=lambda kv: -statistics.median(kv[1])):
            print(f"  {kind:50s} {len(vals):3d} {statistics.median(vals):9.4f} {max(vals):9.4f}")

        allv = [v for vals in by_kind.values() for v in vals]
        print(f"  overall median relative delta: {statistics.median(allv):.4f}")


if __name__ == "__main__":
    main()
