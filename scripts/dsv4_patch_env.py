"""Idempotent patches to .venv-mega2 site-packages for the DeepSeek-V4 Megatron stack.

These are version-skew bugs between ms-swift/Megatron-Core and torch 2.13+cu130. torch 2.13 is
not optional: transformer_engine 2.17 (which mcore-bridge imports unconditionally) links against
a c10 ABI that torch 2.11 does not have, so the stack only imports on 2.13.

Run after any pip install into .venv-mega2:
    .venv-mega2/bin/python scripts/dsv4_patch_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SP = Path(__file__).resolve().parent.parent / ".venv-mega2/lib/python3.11/site-packages"

PATCHES = [
    # torch's _validate_global_plan returns list[str] of errors; ms-swift's stub returns True,
    # meant as "valid" under an older bool API. Truthy now means "errors found", and the caller
    # does '; '.join(True):
    #     TypeError: can only join an iterable
    # An empty list is the correct "no errors" value.
    ("swift/megatron/init.py",
     "    def _validate_global_plan(*args, **kwargs):\n        return True\n",
     "    def _validate_global_plan(*args, **kwargs):\n        return []  # torch>=2.13: list of errors, not a bool\n"),
]


def install_fht_fallback():
    """Provide `fast_hadamard_transform` if the CUDA extension is not installed.

    Megatron's DSA path asserts on it (see scripts/fht_fallback.py). Never overwrite a real
    installation -- the fused kernel is faster than the torch fallback.
    """
    import shutil
    target = SP / "fast_hadamard_transform.py"
    pkg = SP / "fast_hadamard_transform"
    if pkg.exists():
        print("ok      fast_hadamard_transform (CUDA extension installed)")
        return 0
    src = Path(__file__).resolve().parent / "fht_fallback.py"
    if target.exists() and target.read_text() == src.read_text():
        print("ok      fast_hadamard_transform.py fallback (already installed)")
        return 0
    shutil.copy(src, target)
    print("patched fast_hadamard_transform.py (pure-torch FWHT fallback)")
    return 1


def main():
    changed = install_fht_fallback()
    for rel, old, new in PATCHES:
        path = SP / rel
        if not path.exists():
            print(f"MISSING {rel}")
            sys.exit(1)
        text = path.read_text()
        if new in text:
            print(f"ok      {rel} (already patched)")
            continue
        if old not in text:
            print(f"STALE   {rel}: pattern not found -- upstream changed, re-check by hand")
            sys.exit(1)
        path.write_text(text.replace(old, new))
        print(f"patched {rel}")
        changed += 1
    print(f"{changed} patch(es) applied")


if __name__ == "__main__":
    main()
