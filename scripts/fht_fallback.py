"""Pure-PyTorch stand-in for `fast_hadamard_transform`.

Megatron's DeepSeek sparse-attention path calls `rotate_activation` on the Lightning Indexer's
queries and keys, and imports the kernel as a hard dependency:

    megatron/core/transformer/experimental_attention_variant/dsa.py
        try:    from fast_hadamard_transform import hadamard_transform
        except: hadamard_transform = None
        ...
        assert hadamard_transform is not None, "fast_hadamard_transform is not installed."

The upstream package is a CUDA extension whose PyPI sdist ships without its csrc/ directory, so
`pip install fast-hadamard-transform` cannot build. This module is dropped into site-packages as
`fast_hadamard_transform.py` only when the real extension is unavailable, so that same import
succeeds.

It is a genuine fast Walsh-Hadamard transform -- log2(n) vectorised butterfly stages, not an
approximation -- so results match the kernel up to floating-point associativity. It is slower
than the fused kernel, but it runs on `index_head_dim`-wide vectors inside the indexer only, not
on the model's hidden states, so the throughput cost is small.

Install via: .venv-mega2/bin/python scripts/dsv4_patch_env.py
"""

from __future__ import annotations

import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Unnormalised Hadamard transform along the last dim, times `scale`.

    Matches the signature of fast_hadamard_transform.hadamard_transform. The last dimension must
    be a power of two, which holds for DeepSeek-V4's index_head_dim.
    """
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"hadamard_transform needs a power-of-two last dim, got {n}")
    shape = x.shape
    # fp32 accumulation: n=128 sums 128 terms of similar magnitude, and bf16 has 8 mantissa bits.
    v = x.reshape(-1, n).to(torch.float32)
    h = 1
    while h < n:
        # Each contiguous 2h-block splits into halves [i:i+h] and [i+h:i+2h]; stacking the
        # sum/difference back along the same axis restores the original element order.
        v = v.view(-1, n // (2 * h), 2, h)
        a, b = v[:, :, 0, :], v[:, :, 1, :]
        v = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    return (v * scale).reshape(shape).to(x.dtype)


if __name__ == "__main__":
    # Reference check against the explicit Hadamard matrix.
    for n in (2, 8, 128):
        H = torch.ones(1, 1)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        x = torch.randn(5, n)
        got = hadamard_transform(x, scale=n ** -0.5)
        want = x @ H.T * n ** -0.5
        err = (got - want).abs().max().item()
        print(f"n={n:4d} max abs err {err:.2e}")
        assert err < 1e-4
    print("ok")
