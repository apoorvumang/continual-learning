"""Run tau2 with the agent required to consult the knowledge base before acting.

The control this settles: after injection the trained model cut KB_search calls by 74% (21.15 ->
5.44 per episode, p=0.0001) while accuracy moved -0.012, which is statistically nothing but is on
the wrong side of zero. Two incompatible stories fit that:

  the knowledge is fine and it simply stopped looking -- in which case forcing retrieval restores
  accuracy to baseline, and the honest result is "same accuracy at a quarter of the retrieval, with
  a knob to trade one for the other"

  the injected knowledge is subtly wrong and it is now confidently acting on it -- in which case
  forcing retrieval does NOT recover, and the accessibility work planned next would be built on a
  corpus that misleads the model

Nothing downstream is safe to design until this is answered, and it needs no training: same
weights, same tasks, same everything, one added instruction.

tau2 exposes no flag for appending to the agent's system prompt, so patch it here rather than
editing the checkout -- that keeps the manipulation visible in this file instead of buried in a
working-tree diff nobody remembers making.

    python scripts/tau_forced_retrieval.py --port 8000 --tasks 40 --trials 2
"""

from __future__ import annotations

import argparse
import sys

FORCE = """

<retrieval_requirement>
Before you take ANY action that changes state, and before you tell the customer what a policy is,
you MUST call KB_search and read the result. Do this even when you believe you already know the
answer -- your recollection of tool names, thresholds and eligibility rules may be incomplete or
outdated, and the knowledge base is authoritative. Search again whenever the topic shifts.
</retrieval_requirement>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8000")
    ap.add_argument("--model", default="dsv4")
    ap.add_argument("--user-llm", default="openai/deepseek/deepseek-v4-flash-0731")
    ap.add_argument("--retrieval-config", default="bm25")
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--save-to", required=True)
    args = ap.parse_args()

    sys.path.insert(0, "/tmp/tau2-bench/src")
    from tau2.agent import llm_agent

    if "<retrieval_requirement>" not in llm_agent.SYSTEM_PROMPT:
        llm_agent.SYSTEM_PROMPT = llm_agent.SYSTEM_PROMPT + FORCE
    assert "<retrieval_requirement>" in llm_agent.SYSTEM_PROMPT, "patch did not take"
    print("agent system prompt patched with the retrieval requirement", flush=True)

    argv = [
        "tau2", "run",
        "--domain", "banking_knowledge",
        "--retrieval-config", args.retrieval_config,
        "--agent-llm", f"openai/{args.model}",
        "--agent-llm-args", f'{{"api_base":"http://127.0.0.1:{args.port}/v1","api_key":"x"}}',
        "--user-llm", args.user_llm,
        "--num-tasks", str(args.tasks),
        "--num-trials", str(args.trials),
        "--max-concurrency", str(args.concurrency),
        "--save-to", args.save_to,
    ]
    sys.argv = argv
    from tau2.cli import main as tau2_main
    tau2_main()


if __name__ == "__main__":
    main()
