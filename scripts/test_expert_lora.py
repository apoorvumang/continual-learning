"""Correctness tests for expert LoRA. Run before any training run that uses --expert-lora.

    python scripts/test_expert_lora.py

Builds a 4-layer Qwen3.5-MoE (real architecture, tiny dimensions) and checks four things for
every (kernel, mode) combination. Each corresponds to a bug this project actually shipped:

  1. attaching is a bitwise no-op          -- both previous LoRA bugs were silent no-ops that
                                              reported success
  2. the adapter is in the forward path    -- perturbing B must change the logits
  3. the base stays frozen                 -- a LoRA that edits base weights is not a LoRA
  4. merged weights == runtime adapter     -- the serving path must match the training path

(4) is the one that matters most: we train with an adapter and serve merged weights, so a
mismatch means the thing evaluated is not the thing trained. Tolerance is relative, because the
merged path does one matmul where the runtime path does two and adds -- the residual is fp32
round-off at ~1e-6 relative, not error.
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from expert_lora import (  # noqa: E402
    active_implementation, attach_expert_lora, save_expert_lora, verify_identity,
)
from merge_sdf_lora import expand_expert_delta, expert_pairs  # noqa: E402

TINY = dict(num_hidden_layers=4, hidden_size=64, num_experts=8, moe_intermediate_size=32,
            shared_expert_intermediate_size=32, intermediate_size=32, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, vocab_size=128, pad_token_id=0,
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
            linear_value_head_dim=16, num_experts_per_tok=2)


def force_cpu_linear_attention() -> None:
    """Make the gated-delta-net layers use the pure-torch path.

    `layer_types` must contain at least one `linear_attention` (the model does
    `layer_types[::-1].index("linear_attention")`), but when flash-linear-attention is installed
    those layers use it unconditionally -- so this CPU test dies with a CUDA OOM whenever the GPU
    happens to be busy serving something. Nulling the handles makes
    `Qwen3_5MoeGatedDeltaNet.__init__` take the same fallbacks it would in an env without fla.

    `FusedRMSNormGated` is the one that actually bites: it is selected by `is None` rather than by
    `is_fast_path_available`, and it hardcodes `device=torch.cuda.current_device()`, so it
    allocates on the GPU during *construction* however the rest of the model is placed.
    """
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as m
    for attr in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update", "FusedRMSNormGated"):
        setattr(m, attr, None)
    m.is_fast_path_available = False


def tiny_config(impl: str):
    from transformers import AutoConfig
    cfg = copy.deepcopy(AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B").text_config)
    for k, v in TINY.items():
        setattr(cfg, k, v)
    cfg.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    cfg._experts_implementation = impl
    return cfg


def run_case(impl: str, mode: str, tmp: Path) -> tuple[bool, str]:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel
    cfg = tiny_config(impl)
    ids = torch.randint(0, TINY["vocab_size"], (2, 24), generator=torch.Generator().manual_seed(1))

    torch.manual_seed(0)
    plain = Qwen3_5MoeTextModel(cfg).float().eval()
    assert active_implementation(plain.layers[0].mlp.experts) == impl, "kernel not applied"

    runtime = copy.deepcopy(plain)
    for p in runtime.parameters():           # what get_peft_model does in the real trainer
        p.requires_grad_(False)
    summary = attach_expert_lora(runtime, rank=8, alpha=16, mode=mode, dropout=0.0, verbose=False)
    assert summary["experts_implementation"] == impl

    # (1) and (2)
    verify_identity(runtime, {"input_ids": ids})

    frozen = {n: p.detach().clone() for n, p in runtime.named_parameters()
              if ".expert_lora." not in n}
    opt = torch.optim.AdamW([p for p in runtime.parameters() if p.requires_grad], lr=5e-2)
    runtime.train()
    for _ in range(10):
        loss = runtime(input_ids=ids).last_hidden_state.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    runtime.eval()

    # (3)
    after = dict(runtime.named_parameters())
    moved_base = [n for n, v in frozen.items() if not torch.equal(v, after[n].detach())]
    if moved_base:
        return False, f"training changed {len(moved_base)} base tensors, e.g. {moved_base[0]}"

    # (4)
    save_expert_lora(runtime, tmp, summary)
    with torch.no_grad():
        ref = runtime(input_ids=ids).last_hidden_state.clone()
        pre = plain(input_ids=ids).last_hidden_state.clone()

    pairs, read_cfg = expert_pairs(tmp)
    if read_cfg["mode"] != mode:
        return False, f"mode round-tripped as {read_cfg['mode']!r}, expected {mode!r}"
    sd = plain.state_dict()
    for name, pair in pairs.items():
        key = name if name in sd else name.replace("model.", "", 1)
        if key not in sd:
            return False, f"expert tensor {name} has no home in the state dict"
        delta = expand_expert_delta(pair, tuple(sd[key].shape))
        sd[key] = (sd[key].to(torch.float32) + delta).to(sd[key].dtype)
    merged = Qwen3_5MoeTextModel(cfg).float().eval()
    merged.load_state_dict(sd)
    with torch.no_grad():
        got = merged(input_ids=ids).last_hidden_state

    effect = (ref - pre).abs().max().item()
    err = (ref - got).abs().max().item()
    rel = err / max(effect, 1e-30)
    if effect < 1e-3:
        return False, f"adapter barely changed the output ({effect:.2e}) -- suspect a no-op"
    if rel > 1e-4:
        return False, f"merged != runtime: {err:.2e} absolute, {rel:.2e} relative"
    return True, (f"{summary['expert_lora_params']:>8d} params  effect {effect:.3e}  "
                  f"merge rel.err {rel:.1e}")


def main() -> int:
    force_cpu_linear_attention()
    failures = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for impl in ("eager", "grouped_mm"):
            for mode in ("shared", "per-expert"):
                try:
                    ok, detail = run_case(impl, mode, tmp)
                except Exception as exc:  # noqa: BLE001 - report, do not abort the matrix
                    ok, detail = False, f"{type(exc).__name__}: {exc}"
                print(f"  [{'PASS' if ok else 'FAIL'}] {impl:11} {mode:11} {detail}")
                if not ok:
                    failures.append(f"{impl}/{mode}: {detail}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall 4 combinations pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
