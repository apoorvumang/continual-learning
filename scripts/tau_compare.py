"""Compare two tau2 runs: base model vs knowledge-injected, on identical tasks.

Reports three things, because a single average would hide the two ways this experiment can mislead.

pass^k    tau2's own metric. pass^1 is per-trial success; pass^2 is the fraction of tasks solved on
          EVERY trial, which is the honest measure of whether the model reliably knows something
          rather than occasionally guessing it. Injected knowledge should move pass^2 more than
          pass^1 if it is real.

paired    per-task deltas, with a sign test. 40 tasks is small, the variance between tau2 tasks is
          large, and an unpaired mean difference over so few tasks can look convincing while
          resting on two lucky episodes. Pairing removes task difficulty from the comparison.

health    the failure mode that would invalidate the whole thing. Continued pretraining can degrade
          tool-call formatting and reasoning, and tau2 scores a malformed call exactly like a wrong
          answer. If the trained arm regresses, this separates "forgot how to act" from "still does
          not know the policy" -- which are opposite conclusions about the method.

    python scripts/tau_compare.py --base /tmp/tau_base_40x2.json --trained /tmp/tau_trained_40x2.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load(p: str) -> list:
    f = Path(p)
    if f.is_dir():
        f = f / "results.json"
    d = json.loads(f.read_text())
    return d.get("simulations") or []


def by_task(sims: list) -> dict:
    out = defaultdict(list)
    for s in sims:
        r = (s.get("reward_info") or {}).get("reward")
        if r is not None:
            out[s["task_id"]].append(float(r))
    return out


def pass_hat_k(runs: dict, k: int) -> float:
    """Fraction of tasks solved on at least k trials -- k = n_trials is the all-trials case."""
    ok = [t for t, rs in runs.items() if len(rs) >= k and sum(1 for r in rs if r >= 1.0) >= k]
    return len(ok) / max(len(runs), 1)


def health(sims: list) -> dict:
    """Did the agent still behave like an agent?"""
    n_calls = n_msgs = n_empty = 0
    steps, errs = [], 0
    for s in sims:
        msgs = s.get("messages") or []
        a = [m for m in msgs if m.get("role") == "assistant"]
        steps.append(len(a))
        for m in a:
            n_msgs += 1
            tc = m.get("tool_calls") or []
            n_calls += len(tc)
            if not tc and not (m.get("content") or "").strip():
                n_empty += 1
        if s.get("termination_reason") not in (None, "user_stop", "agent_stop"):
            errs += 1
    return {"assistant_turns": n_msgs, "tool_calls": n_calls,
            "calls_per_turn": n_calls / max(n_msgs, 1),
            "empty_turns": n_empty / max(n_msgs, 1),
            "mean_steps": sum(steps) / max(len(steps), 1),
            "odd_termination": errs / max(len(sims), 1)}


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--label-base", default="base")
    ap.add_argument("--label-trained", default="trained")
    args = ap.parse_args()

    bs, ts = load(args.base), load(args.trained)
    b, t = by_task(bs), by_task(ts)
    shared = sorted(set(b) & set(t))
    print(f"{len(bs)} vs {len(ts)} simulations; {len(shared)} tasks in both\n")
    if not shared:
        print("no shared tasks -- were the runs configured with the same --num-tasks?")
        return

    kmax = max(1, min(min(len(b[x]) for x in shared), min(len(t[x]) for x in shared)))
    print(f"{'metric':22} {args.label_base:>9} {args.label_trained:>9} {'delta':>9}")
    bm = sum(sum(b[x]) / len(b[x]) for x in shared) / len(shared)
    tm = sum(sum(t[x]) / len(t[x]) for x in shared) / len(shared)
    print(f"{'avg reward':22} {bm:9.3f} {tm:9.3f} {tm-bm:+9.3f}")
    for k in range(1, kmax + 1):
        bk = pass_hat_k({x: b[x] for x in shared}, k)
        tk = pass_hat_k({x: t[x] for x in shared}, k)
        print(f"{'pass^' + str(k):22} {bk:9.3f} {tk:9.3f} {tk-bk:+9.3f}")

    wins = losses = ties = 0
    movers = []
    for x in shared:
        bb, tt = sum(b[x]) / len(b[x]), sum(t[x]) / len(t[x])
        if tt > bb:
            wins += 1
        elif tt < bb:
            losses += 1
        else:
            ties += 1
        if tt != bb:
            movers.append((tt - bb, x))
    p = sign_test(wins, losses)
    print(f"\nper-task: {wins} better, {losses} worse, {ties} unchanged   sign test p={p:.4f}")
    movers.sort(reverse=True)
    for d, x in movers[:5]:
        print(f"   +{d:.2f}  {x}")
    for d, x in movers[-5:][::-1]:
        if d < 0:
            print(f"   {d:.2f}  {x}")

    hb, ht = health(bs), health(ts)
    print(f"\n{'agent health':22} {args.label_base:>9} {args.label_trained:>9}")
    for k in ("calls_per_turn", "empty_turns", "mean_steps", "odd_termination"):
        print(f"{k:22} {hb[k]:9.3f} {ht[k]:9.3f}")
    if ht["calls_per_turn"] < hb["calls_per_turn"] * 0.8 or ht["empty_turns"] > hb["empty_turns"] * 2:
        print("\n  WARNING: the trained model calls tools noticeably less. A reward drop here is\n"
              "  degraded agent behaviour, not absent knowledge -- read the traces before concluding\n"
              "  anything about knowledge injection.")


if __name__ == "__main__":
    main()
