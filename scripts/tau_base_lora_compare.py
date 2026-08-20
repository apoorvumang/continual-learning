"""Three-way comparison on golden_retrieval: base model, instruct-trained LoRA, base-trained LoRA.

The question is where a knowledge LoRA should be fit. Training the instruct checkpoint directly on
documents measurably damages its ability to act -- pooled over 93 tasks x 4 trials, base scores 0.583
and the Active Reading arm 0.478 (sign test p=0.0026, paired t=-3.83) -- and the failure profile is
specific: calls passing a dict where the API wants a JSON string, and attempts to call tool names the
model half-invented. Both read as damage to instruction-tuned machinery rather than missing facts.

So this arm fits the identical corpus against DeepSeek-V4-Flash-Base and transplants the adapter into
the 0731 instruct checkpoint. Same data, same recipe, same evaluation; the only variable is which
checkpoint the gradient steps were taken against.

Reports reward, the paired test against each reference, and the two mechanism numbers that diagnose
the regression: action_match and the tool-error profile.

    python scripts/tau_base_lora_compare.py
"""

from __future__ import annotations

import collections
import json
import math
import os
import statistics
from collections import defaultdict

ARMS = [
    ("base (untrained)", ['/tmp/tau_base_golden_97x2.json', '/tmp/tau_base_golden_97x2_r2.json']),
    ("LoRA on instruct", ['/tmp/tau_ar_golden_97x2.json', '/tmp/tau_ar_golden_97x2_r2.json']),
    ("LoRA on base -> instruct", ['/tmp/tau_baselora_golden_97x2.json']),
]


def sims(paths):
    out = []
    for p in paths:
        f = p + '/results.json'
        if os.path.exists(f):
            out += json.load(open(f))['simulations']
    return out


def by_task(s):
    t = defaultdict(list)
    for x in s:
        r = (x.get('reward_info') or {}).get('reward')
        if r is not None:
            t[x['task_id']].append(float(r))
    return t


def health(s):
    ac = ok = 0
    errs = collections.Counter()
    steps = []
    for x in s:
        for a in (x.get('reward_info') or {}).get('action_checks') or []:
            ac += 1
            ok += 1 if a.get('action_match') else 0
        msgs = x.get('messages') or []
        steps.append(len([m for m in msgs if m.get('role') == 'assistant']))
        for m in msgs:
            if m.get('role') == 'tool':
                c = str(m.get('content', ''))
                if c.startswith('Error') or 'Error:' in c[:60]:
                    errs[c[:72]] += 1
    return (ok / max(ac, 1), ac, statistics.mean(steps) if steps else 0, errs)


def sign(a, b, shared):
    d = [statistics.mean(a[k]) - statistics.mean(b[k]) for k in shared]
    w = sum(1 for x in d if x > 0)
    l = sum(1 for x in d if x < 0)
    n = w + l
    p = (min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else 1.0)
    t = (statistics.mean(d) / (statistics.stdev(d) / math.sqrt(len(d)))
         if len(d) > 1 and statistics.stdev(d) > 0 else 0.0)
    return statistics.mean(d), w, l, len(shared) - w - l, p, t


def main():
    loaded = [(n, sims(p)) for n, p in ARMS]
    have = [(n, s) for n, s in loaded if s]
    for n, s in loaded:
        if not s:
            print(f"({n}: no results yet)")
    if len(have) < 2:
        return
    tasks = [by_task(s) for _n, s in have]
    shared = sorted(set.intersection(*[set(t) for t in tasks]))
    print(f"\ngolden_retrieval, paired over {len(shared)} tasks\n")
    print(f"{'arm':28} {'sims':>5} {'reward':>7} {'action_match':>13} {'steps':>6}")
    for (n, s), t in zip(have, tasks):
        hm, _ac, st, _e = health(s)
        r = statistics.mean(statistics.mean(t[k]) for k in shared)
        print(f"{n:28} {len(s):5d} {r:7.3f} {hm:12.1%} {st:6.1f}")

    ref = dict(zip([n for n, _ in have], tasks))
    if "LoRA on base -> instruct" in ref:
        new = ref["LoRA on base -> instruct"]
        for other in ("base (untrained)", "LoRA on instruct"):
            if other in ref:
                d, w, l, ti, p, t = sign(new, ref[other], shared)
                print(f"\n  vs {other:26} delta {d:+.3f}  {w}W/{l}L/{ti}T  "
                      f"sign p={p:.4f}  paired t={t:+.2f}")

    print("\ntool errors (the regressions this arm is meant to avoid):")
    for n, s in have:
        _hm, _ac, _st, errs = health(s)
        dict_err = sum(v for k, v in errs.items() if 'must be str, bytes' in k)
        unlocked = sum(v for k, v in errs.items() if 'has not been unlock' in k)
        print(f"  {n:28} total {sum(errs.values()):5d} | dict-not-string {dict_err:4d} | "
              f"not-unlocked {unlocked:4d}")


if __name__ == "__main__":
    main()
