"""Print the linear-layer names of Qwen3.5 so we know what LoRA can target."""

import collections

import torch
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen3.5-9B-Base"

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cpu")
print(type(model).__name__)

names = collections.Counter()
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        # strip layer indices so we see the shapes of the repeated blocks
        parts = [p for p in name.split(".") if not p.isdigit()]
        names[".".join(parts)] += 1

for name, count in names.most_common():
    print(f"{count:4d}  {name}")

total = sum(p.numel() for p in model.parameters())
text = sum(p.numel() for n, p in model.named_parameters() if "visual" not in n)
emb = sum(p.numel() for n, p in model.named_parameters() if "embed_tokens" in n or "lm_head" in n)
print(f"\ntotal {total/1e9:.2f}B  text-only {text/1e9:.2f}B  embed+head {emb/1e9:.2f}B")
