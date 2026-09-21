"""
Use GPT-5.4 as behaviour judge. Given a generated CoT and a target behaviour,
return a 0..5 intensity score and extract the model's final short answer for
accuracy comparison.

Usage:
  python judge_behaviour.py --gens results/steered_generations.json \
                            --eval_problems data/eval_problems.json \
                            --out results/judgments.json
"""
import os, json, argparse, sys, time, re, concurrent.futures
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
    os.environ.pop(k, None)
os.environ["no_proxy"] = "*"

from openai import OpenAI

API_KEY  = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL    = "gpt-5.4"

BEHAVIOUR_DESCRIPTIONS = {
    "uncertainty": (
        "expressing UNCERTAINTY / HEDGING — phrases like 'not sure', 'maybe', "
        "'I think', 'possibly', 'let me double-check', 'I could be wrong'"
    ),
    "example_generation": (
        "generating a CONCRETE VALIDATION EXAMPLE — plugging in specific values, "
        "trying a small case (n=3 etc.), sanity-checking with a specific instance"
    ),
    "backtracking": (
        "BACKTRACKING or self-correction — starting one approach, realising it "
        "is wrong, and switching direction. Phrases like 'wait, that's wrong', "
        "'actually, no', 'let me reconsider', 'hmm, that doesn't work'"
    ),
}

JUDGE_PROMPT = """You are rating a chain-of-thought produced by a reasoning LLM.

Target behaviour: {behaviour}
Definition: {definition}

Instructions:
1. Rate how strongly the behaviour is present on a 0-5 scale:
   0 = fully absent, 5 = extremely and repeatedly present
2. Extract the final short answer the model gave (if any); use "NO_ANSWER" if unclear.
3. Judge whether the final answer is correct given the reference: {reference}
   (Compare on meaning, not just literal string; accept equivalent numeric forms.)
4. Judge overall coherence of the reasoning on a 0-5 scale
   (0 = incoherent garbage, 5 = clean).

Problem: {problem}

Chain of thought:
<<<
{cot}
>>>

Return STRICTLY a compact JSON object, no code fence, no commentary:
{{"intensity": <int 0-5>, "final_answer": "<str>", "correct": <true|false>, "coherence": <int 0-5>}}
"""


def robust_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"): text = text[4:]
        text = text.strip()
    s = text.find("{"); e = text.rfind("}")
    return json.loads(text[s:e+1])


def judge_one(client, problem, cot, behaviour, reference):
    msg = JUDGE_PROMPT.format(
        behaviour=behaviour,
        definition=BEHAVIOUR_DESCRIPTIONS[behaviour],
        problem=problem,
        cot=cot[:6000],
        reference=reference,
    )
    for k in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": msg}],
                max_tokens=400,
            )
            d = robust_json(r.choices[0].message.content)
            for f in ("intensity", "coherence"):
                d[f] = int(d[f])
            d["correct"] = bool(d["correct"])
            return d
        except Exception as e:
            print(f"[judge retry {k+1}/3] {e}", file=sys.stderr)
            time.sleep(1)
    return {"intensity": -1, "final_answer": "JUDGE_FAIL", "correct": False, "coherence": -1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--eval_problems", default="data/eval_problems.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    with open(args.gens) as f:
        gens_obj = json.load(f)
    generations = gens_obj["generations"] if isinstance(gens_obj, dict) and "generations" in gens_obj else gens_obj

    with open(args.eval_problems) as f:
        problems = json.load(f)
    p2ref = {p["problem"]: p["answer"] for p in problems}

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # If any generation has behaviour="none" (a true no-directive baseline),
    # judge that generation once per real behaviour so we can compare to steering α=0.
    expanded = []
    for item in generations:
        if item.get("behaviour") == "none":
            for b in BEHAVIOUR_DESCRIPTIONS:
                expanded.append({**item, "behaviour": b, "condition": "baseline"})
        else:
            expanded.append(item)

    def work(item):
        ref = p2ref.get(item["problem"], "unknown")
        j = judge_one(client, item["problem"], item["text"], item["behaviour"], ref)
        return {**item, **{"judge_" + k: v for k, v in j.items()}, "reference": ref}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, r in enumerate(pool.map(work, expanded)):
            results.append(r)
            if (i+1) % 10 == 0:
                print(f"[judge] {i+1}/{len(expanded)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"steer_layers": gens_obj.get("steer_layers"), "alphas": gens_obj.get("alphas"), "judgments": results}, f, indent=2)
    print(f"[save] wrote {args.out}")


if __name__ == "__main__":
    main()
