"""Is the weak `direct` score real, or an artifact of the sampling config?

Qwen3.5's card recommends presence_penalty=1.5 for non-thinking mode. That is a very
high penalty for short factual answers -- it pushes the model off tokens it has already
used, which is exactly what a correct "X died on <date>" answer needs to repeat. This
compares configs on pre-cutoff events the model got RIGHT under MCQ but WRONG under
direct, so any config difference is visible.
"""

import json
import sys

import openai

BASE = "http://127.0.0.1:8011/v1"
SYSTEM = ("Answer the question directly and factually based on what you know. "
          "If you are not sure, say so, but give your best answer.")

CONFIGS = {
    "card-nonthink (pp1.5,t0.7)": dict(temperature=0.7, top_p=0.8, presence_penalty=1.5,
                                       extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}),
    "nonthink pp0 (t0.7)":        dict(temperature=0.7, top_p=0.8, presence_penalty=0.0,
                                       extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}),
    "nonthink greedy (pp0,t0)":   dict(temperature=0.0, presence_penalty=0.0,
                                       extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
    "card-thinking (pp1.5,t1.0)": dict(temperature=1.0, top_p=0.95, presence_penalty=1.5,
                                       extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}),
}


def main():
    run = {json.loads(l)["event_id"]: json.loads(l)
           for l in open(sys.argv[1] + "/runs/qwen3.5-9b__direct.jsonl")}
    graded_d = {json.loads(l)["event_id"]: json.loads(l)
                for l in open(sys.argv[1] + "/graded/qwen3.5-9b__direct.jsonl")}
    graded_m = {json.loads(l)["event_id"]: json.loads(l)
                for l in open(sys.argv[1] + "/graded/qwen3.5-9b__mcq.jsonl")}
    events = {json.loads(l)["id"]: json.loads(l)
              for l in open(sys.argv[1] + "/data/events.jsonl")}

    # pre-2025 events: MCQ correct but direct wrong -> the interesting disagreements
    picks = [eid for eid, g in graded_d.items()
             if g["month"] < "2025-01" and g["label"] == "incorrect"
             and graded_m.get(eid, {}).get("label") == "correct"][:6]
    print(f"{len(picks)} disagreement cases (MCQ correct, direct incorrect, pre-2025)\n")

    client = openai.OpenAI(api_key="x", base_url=BASE)
    for eid in picks:
        ev = events[eid]
        print("=" * 100)
        print(f"[{ev['month']}] {ev['question_direct']}")
        print(f"  TRUTH: {ev['expected_direct']}")
        for name, cfg in CONFIGS.items():
            r = client.chat.completions.create(
                model="qwen3.5-9b", max_completion_tokens=2048,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": ev["question_direct"]}],
                **cfg)
            msg = r.choices[0].message
            txt = " ".join((msg.content or "").split())
            think = getattr(msg, "reasoning_content", None)
            tag = f" [thought {len(think)}ch]" if think else ""
            print(f"  {name:28s}{tag}: {txt[:220]}")
        print()


if __name__ == "__main__":
    main()
