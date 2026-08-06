"""LoRA that reaches the routed experts -- the 92% of an MoE that PEFT cannot touch.

Why this file exists. In Qwen3.5-MoE (and DeepSeek-V4, and every recent MoE) the 256 experts of
a layer are stored as two fused 3D parameters:

    mlp.experts.gate_up_proj   (num_experts, 2 * moe_intermediate, hidden)
    mlp.experts.down_proj      (num_experts, hidden, moe_intermediate)

PEFT injects adapters by walking `named_modules()` and replacing `nn.Linear` instances. A 3D
`nn.Parameter` is not a module, so there is nothing to replace and `target_modules=["gate_up_proj"]`
raises "Target modules not found". Worse, the way we actually hit it: our TARGETS list contains
`down_proj`, which DOES match the shared expert, so the call succeeded while silently skipping
every routed expert. Measured on arm P -- 250 of 1811 tensors changed, and the routed experts
(33.02B of 35.95B params, 91.8%) were byte-identical to stock.

None of that is a limitation of autograd. `self.gate_up_proj[expert_idx]` is an ordinary
indexing op and gradients flow through it fine (verified: 8/8 fused expert tensors receive
nonzero grads from a plain backward). The gap was purely PEFT plumbing.

How this works. Rather than materialising `W + BA` for all 256 experts -- which would add 66 GB
on the 35B model -- the delta is applied activation-side, inside the expert loop, so only the
experts a token actually routes to ever compute one. Two modes:

    shared       one (A, B) per fused tensor, shared by all experts in that layer.
                 7.2M params at r=32 on the 35B: the same order as the attention adapter, and
                 every expert gets gradient from every token. Start here.
    per-expert   one (A, B) per expert: 1.85B params at r=32, 48x arm P's adapter. More
                 capacity, but with top-6-of-256 routing each expert sees only ~2.3% of tokens,
                 so each adapter gets ~44x sparser gradient signal than the shared variant.

B is zero-initialised, so an untrained attachment is a bitwise no-op -- `verify_identity()`
asserts this, because a silent no-op is exactly how the last two LoRA bugs presented.

Usage -- attach AFTER get_peft_model, which freezes everything it finds:

    model = get_peft_model(model, LoraConfig(...))
    attach_expert_lora(model, rank=32, alpha=64, mode="shared")
    ...
    save_expert_lora(model, out_dir)          # peft's save_pretrained drops these
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers.integrations.moe import _grouped_linear, grouped_mm_experts_forward

MODES = ("shared", "per-expert")


class FusedExpertLoRA(nn.Module):
    """Low-rank delta for one fused `(num_experts, out, in)` expert weight.

    Parameters are named `A` / `B` rather than the conventional `lora_A` / `lora_B` on purpose:
    peft's `get_peft_model_state_dict` keeps every key containing the substring `"lora_"`, so
    conventional names here would be silently absorbed into the peft adapter file and then fail
    to load back as peft adapters. `expert_lora.gate_up.A` contains `"lora."`, not `"lora_"`.
    """

    def __init__(self, num_experts: int, in_features: int, out_features: int, rank: int,
                 alpha: float, mode: str, dropout: float, dtype, device):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.per_expert = mode == "per-expert"
        shape_a = (num_experts, rank, in_features) if self.per_expert else (rank, in_features)
        shape_b = (num_experts, out_features, rank) if self.per_expert else (out_features, rank)
        self.A = nn.Parameter(torch.empty(shape_a, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.zeros(shape_b, dtype=dtype, device=device))
        # Same init as peft: kaiming-uniform on A, zeros on B, so the delta starts at exactly 0.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor, expert_idx) -> torch.Tensor:
        """Delta for the tokens routed to `expert_idx`: (tokens, in) -> (tokens, out)."""
        a = self.A[expert_idx] if self.per_expert else self.A
        b = self.B[expert_idx] if self.per_expert else self.B
        return F.linear(F.linear(self.drop(x), a), b) * self.scaling

    def delta(self, expert_idx: int | None = None) -> torch.Tensor:
        """The explicit weight delta, for merging. `(out, in)`."""
        a = self.A[expert_idx] if self.per_expert else self.A
        b = self.B[expert_idx] if self.per_expert else self.B
        return self.scaling * (b.to(torch.float32) @ a.to(torch.float32))


def _is_fused_experts(module: nn.Module) -> bool:
    """A module holding fused 3D expert weights, as Qwen3.5-MoE and DeepSeek-V4 both do."""
    gu = getattr(module, "gate_up_proj", None)
    dn = getattr(module, "down_proj", None)
    return (isinstance(gu, nn.Parameter) and gu.ndim == 3
            and isinstance(dn, nn.Parameter) and dn.ndim == 3
            and hasattr(module, "act_fn"))


def _grouped_delta(lora: FusedExpertLoRA, x_g: torch.Tensor, offsets) -> torch.Tensor:
    """LoRA delta for expert-sorted rows `x_g` (S, in) -> (S, out).

    In per-expert mode the rows are already sorted by expert and `offsets` marks the group
    boundaries, so the two low-rank matmuls are themselves grouped matmuls -- the same kernel
    the base weights use. In shared mode every row uses the same (A, B), so it is two dense
    matmuls and the expert structure is irrelevant.
    """
    x = lora.drop(x_g)
    if lora.per_expert:
        return _grouped_linear(_grouped_linear(x, lora.A, offsets), lora.B, offsets) * lora.scaling
    return F.linear(F.linear(x, lora.A), lora.B) * lora.scaling


def _patched_grouped_forward(self, hidden_states, top_k_index, top_k_weights):
    """`grouped_mm_experts_forward` with three lines added for the LoRA delta.

    A faithful copy of upstream, so a change in transformers shows up as a diff here rather
    than as silently wrong numerics. `_check_upstream` guards the markers this relies on.
    """
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    selected_hidden_states = hidden_states[token_idx]

    perm = torch.argsort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)

    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    selected_hidden_states_g = selected_hidden_states[perm]

    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    lora = self.expert_lora
    selected_weights = self.gate_up_proj if self.has_gate else self.up_proj
    selected_biases = None
    if self.has_bias:
        bias = self.gate_up_proj_bias if self.has_gate else self.up_proj_bias
        selected_biases = bias[expert_ids_g]

    proj_out = _grouped_linear(selected_hidden_states_g, selected_weights, offsets,
                               bias=selected_biases, is_transposed=self.is_transposed)
    proj_out = proj_out + _grouped_delta(lora.gate_up, selected_hidden_states_g, offsets)  # added

    proj_out = self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)
    activated_g = proj_out                                                                # added

    selected_biases = self.down_proj_bias[expert_ids_g] if self.has_bias else None
    proj_out = _grouped_linear(proj_out, self.down_proj, offsets,
                               bias=selected_biases, is_transposed=self.is_transposed)
    proj_out = proj_out + _grouped_delta(lora.down, activated_g, offsets)                  # added

    weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
    weighted_out = weighted_out[inv_perm]
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    return final_hidden_states.to(hidden_states.dtype)


def _patched_eager_forward(self, hidden_states, top_k_index, top_k_weights):
    """Upstream `Qwen3_5MoeExperts.forward` (the eager loop) with two lines added."""
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    lora = self.expert_lora
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate_up = F.linear(current_state, self.gate_up_proj[expert_idx])
        gate_up = gate_up + lora.gate_up(current_state, expert_idx)            # <-- added
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        out = F.linear(current_hidden_states, self.down_proj[expert_idx])
        out = out + lora.down(current_hidden_states, expert_idx)               # <-- added
        out = out * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, out.to(final_hidden_states.dtype))

    return final_hidden_states


# Which upstream function each patch was derived from, and the markers that must still be present.
_PATCHES = {
    "grouped_mm": (_patched_grouped_forward, grouped_mm_experts_forward,
                   ("_grouped_linear(", "_apply_gate", "inv_perm", "torch.cumsum")),
    "eager": (_patched_eager_forward, None,
              ("gate_up_proj[expert_idx]", "down_proj[expert_idx]",
               "chunk(2, dim=-1)", "index_add_")),
}


def active_implementation(module: nn.Module) -> str:
    """Which experts kernel this module will actually run.

    transformers dispatches `Qwen3_5MoeExperts.forward` through `ExpertsInterface`, defaulting
    to `grouped_mm` -- NOT the eager loop in the modelling file. Patching the eager loop alone
    would silently replace a fused grouped matmul with a Python loop over 256 experts per layer.
    """
    impl = getattr(getattr(module, "config", None), "_experts_implementation", None)
    return impl or "eager"


def _check_upstream(module: nn.Module, impl: str) -> None:
    """Fail loudly if the upstream function a patch was derived from has changed."""
    import inspect
    patched, upstream, markers = _PATCHES[impl]
    src = inspect.getsource(upstream if upstream is not None else type(module).forward)
    for marker in markers:
        if marker not in src:
            raise RuntimeError(
                f"the upstream {impl} experts forward no longer contains {marker!r}. The patch in "
                f"scripts/expert_lora.py was derived from a different version and would produce "
                f"silently wrong numerics. Re-derive it from the current source before training.")


class _ExpertLoRAPair(nn.Module):
    """The two adapters for one experts module, so they show up in `named_parameters()`."""

    def __init__(self, gate_up: FusedExpertLoRA, down: FusedExpertLoRA):
        super().__init__()
        self.gate_up = gate_up
        self.down = down


def attach_expert_lora(model: nn.Module, rank: int = 32, alpha: float = 64,
                       mode: str = "shared", dropout: float = 0.05,
                       verbose: bool = True) -> dict:
    """Attach LoRA to every fused-3D expert tensor in `model`. Returns a summary dict.

    Call this AFTER `get_peft_model`: peft sets `requires_grad=False` on everything it can see,
    so an adapter attached beforehand would be frozen and train nothing.
    """
    found = [(n, m) for n, m in model.named_modules() if _is_fused_experts(m)]
    if not found:
        raise RuntimeError(
            "no fused 3D expert tensors found -- this model does not need expert LoRA (its "
            "experts are probably ordinary nn.Linear, which peft already reaches).")

    impls = {active_implementation(m) for _, m in found}
    if len(impls) != 1:
        raise RuntimeError(f"mixed experts implementations across layers: {impls}")
    impl = impls.pop()
    if impl not in _PATCHES:
        raise RuntimeError(
            f"experts_implementation={impl!r} has no expert-LoRA patch (have "
            f"{sorted(_PATCHES)}). Load the model with experts_implementation='grouped_mm' or "
            f"'eager', or derive a patch for {impl!r} from transformers/integrations/moe.py.")

    added = 0
    for name, mod in found:
        if hasattr(mod, "expert_lora"):
            raise RuntimeError(f"{name} already has expert LoRA attached")
        _check_upstream(mod, impl)
        e, out_gu, in_gu = mod.gate_up_proj.shape
        _, out_dn, in_dn = mod.down_proj.shape
        kw = dict(rank=rank, alpha=alpha, mode=mode, dropout=dropout,
                  dtype=mod.gate_up_proj.dtype, device=mod.gate_up_proj.device)
        mod.expert_lora = _ExpertLoRAPair(
            FusedExpertLoRA(e, in_gu, out_gu, **kw),
            FusedExpertLoRA(e, in_dn, out_dn, **kw))
        # bind the patched forward to this instance only, matching the active kernel
        mod.forward = _PATCHES[impl][0].__get__(mod, type(mod))
        added += sum(p.numel() for p in mod.expert_lora.parameters())

    for _, mod in found:
        for p in mod.expert_lora.parameters():
            p.requires_grad_(True)

    n_experts = found[0][1].gate_up_proj.shape[0]
    summary = {"mode": mode, "rank": rank, "alpha": alpha, "dropout": dropout,
               "layers_patched": len(found), "num_experts": n_experts,
               "expert_lora_params": added, "experts_implementation": impl}
    if verbose:
        print(f"expert LoRA [{mode}] r={rank} alpha={alpha}: {len(found)} layers x "
              f"{n_experts} experts, {added/1e6:.1f}M trainable params "
              f"(kernel: {impl})")
    return summary


def expert_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if ".expert_lora." in k}


def save_expert_lora(model: nn.Module, out_dir: str | Path, summary: dict) -> Path:
    """peft's `save_pretrained` keeps only keys containing `lora_`, so save these ourselves."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = expert_lora_state_dict(model)
    if not sd:
        raise RuntimeError("no expert-LoRA tensors in the model state dict")
    save_file(sd, str(out / "expert_lora.safetensors"))
    (out / "expert_lora_config.json").write_text(json.dumps(summary, indent=1))
    return out / "expert_lora.safetensors"


def load_expert_lora(model: nn.Module, adapter_dir: str | Path) -> int:
    """Load saved expert adapters into a model that already has them attached."""
    d = Path(adapter_dir)
    sd = load_file(str(d / "expert_lora.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    stray = [k for k in unexpected if ".expert_lora." in k]
    if stray:
        raise RuntimeError(f"expert adapter keys not present in model: {stray[:3]}")
    return len(sd)


@torch.no_grad()
def verify_identity(model: nn.Module, sample_input: dict, atol: float = 0.0) -> None:
    """Assert a freshly attached expert LoRA changes nothing (B is zero-initialised).

    Worth doing every time. Both LoRA bugs in this project so far presented as a silent no-op
    that reported success, so "did attaching change the output" and "did training change the
    output" have to be checked separately.
    """
    def out_of(m):
        o = m(**sample_input)
        return (o.logits if getattr(o, "logits", None) is not None
                else o.last_hidden_state).float()

    was = model.training
    model.eval()
    ref = out_of(model).clone()
    for m in model.modules():
        if hasattr(m, "expert_lora"):
            for lora in (m.expert_lora.gate_up, m.expert_lora.down):
                nn.init.normal_(lora.B, std=0.02)          # break the zero-init
    now = out_of(model)
    if torch.allclose(ref, now, atol=1e-6):
        raise RuntimeError(
            "perturbing the expert LoRA did not change the logits -- the adapter is a no-op and "
            "is not in the forward path.")
    for m in model.modules():
        if hasattr(m, "expert_lora"):
            for lora in (m.expert_lora.gate_up, m.expert_lora.down):
                nn.init.zeros_(lora.B)                      # restore
    back = out_of(model)
    max_diff = (ref - back).abs().max().item()
    if max_diff > atol:
        raise RuntimeError(f"zero-initialised expert LoRA is not a no-op: max |diff| {max_diff}")
    model.train(was)
    print(f"expert LoRA verified: in the forward path, and a bitwise no-op at init "
          f"(max |diff| {max_diff})")
