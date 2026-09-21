"""
Generate contrastive prompt/response pairs for three reasoning behaviours:
  - uncertainty (hedging)
  - example_generation (validation example)
  - backtracking (self-correction)

Each pair shares an identical *problem prefix* and differs only in the *continuation*:
  - "present": the continuation exhibits the target behaviour
  - "absent" : the continuation does not

We use GPT-5.4 via the DMX API to generate diverse reasoning problems and both
continuations. Output: data/contrastive_pairs.json
"""
import os, json, sys, time, argparse, random
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
    os.environ.pop(k, None)
os.environ["no_proxy"] = "*"; os.environ["NO_PROXY"] = "*"

from openai import OpenAI

API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "gpt-5.4"

BEHAVIOUR_INSTRUCTIONS = {
    "uncertainty": (
        "The 'present' continuation must express UNCERTAINTY / HEDGING — using phrases such as "
        "'I'm not entirely sure', 'this might be wrong', 'I could be mistaken', 'let me double check', "
        "'possibly', 'I think but am not certain', 'maybe', 'perhaps'. It should still be a legitimate "
        "step in solving the problem. "
        "The 'absent' continuation must be CONFIDENT and ASSERTIVE — using phrases such as "
        "'clearly', 'obviously', 'the answer is definitively', 'we know for certain', 'the exact value is'. "
        "It should perform the same reasoning step but with confident phrasing."
    ),
    "example_generation": (
        "The 'present' continuation must GENERATE A CONCRETE VALIDATION EXAMPLE — for instance "
        "'let me try a small case, say n=3', 'let me plug in x=2 and check', 'let me test with the "
        "specific value...', 'consider the concrete case where...'. It should actually work through a small example. "
        "The 'absent' continuation must proceed with ABSTRACT / GENERAL reasoning without "
        "testing a concrete case — using symbolic manipulation, direct formula application, or "
        "general argument only."
    ),
    "backtracking": (
        "The 'present' continuation must BACKTRACK — start one line of reasoning, realise it is "
        "wrong or a dead end, and switch approach. Use phrases such as 'wait, that's wrong', "
        "'actually, that doesn't work', 'hmm, let me reconsider', 'no, I made a mistake', "
        "'on second thought, let me try a different approach'. "
        "The 'absent' continuation must proceed LINEARLY without backtracking — a single clean "
        "chain of reasoning without any self-correction or approach change."
    ),
}

PROMPT_TEMPLATE = """You are generating contrastive continuations for a reasoning-behaviour dataset.

I will give you a reasoning problem and its opening chain-of-thought prefix. You must
produce TWO alternative NEXT-CONTINUATIONS (about 30–60 words each) that share the same
underlying reasoning step but differ ONLY in the target behavioural dimension.

Target behaviour: {behaviour}

{instructions}

Reasoning problem: {problem}
Opening chain-of-thought prefix: {prefix}

Return STRICTLY valid JSON with two keys, no code fence, no commentary:
{{"present": "<continuation exhibiting the behaviour>", "absent": "<continuation without the behaviour>"}}
"""

PROBLEM_GEN_PROMPT = """Generate {n} short, diverse reasoning problems. Each problem should be
solvable by a step-by-step chain of thought and fall into one of these categories: arithmetic
word problem, algebra, geometry, combinatorics, elementary number theory, simple logic puzzle,
simple physics, simple probability. Vary difficulty from easy to medium.

For each problem, also produce a short opening chain-of-thought prefix (~15-25 words) that
begins solving it but stops BEFORE the next natural reasoning step — so a continuation can
be attached.

Return STRICTLY a JSON list of {n} objects, no code fence, no commentary. Each object has
keys "problem" and "prefix"."""


def robust_json(text: str):
    """Try hard to parse JSON out of a possibly-messy model reply."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    # find first { or [
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        s = text.find(open_ch)
        if s >= 0:
            e = text.rfind(close_ch)
            if e > s:
                try:
                    return json.loads(text[s:e+1])
                except json.JSONDecodeError:
                    pass
    return json.loads(text)


def generate_problems(client, n):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROBLEM_GEN_PROMPT.format(n=n)}],
        max_tokens=4000,
    )
    return robust_json(r.choices[0].message.content)


def generate_pair(client, behaviour, problem, prefix, retries=3):
    msg = PROMPT_TEMPLATE.format(
        behaviour=behaviour,
        instructions=BEHAVIOUR_INSTRUCTIONS[behaviour],
        problem=problem,
        prefix=prefix,
    )
    for k in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": msg}],
                max_tokens=800,
            )
            d = robust_json(r.choices[0].message.content)
            assert "present" in d and "absent" in d, f"missing keys: {d.keys()}"
            assert isinstance(d["present"], str) and isinstance(d["absent"], str)
            assert len(d["present"].split()) >= 8 and len(d["absent"].split()) >= 8
            return d
        except Exception as e:
            print(f"[retry {k+1}/{retries}] {behaviour}: {e}", file=sys.stderr)
            time.sleep(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_problems", type=int, default=50)
    ap.add_argument("--out", default="/data/zhenqian/Reproduction1/cc/reasoning/thinking_reasoning_steering/data/contrastive_pairs.json")
    args = ap.parse_args()

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"[gen] generating {args.n_problems} problems...")
    problems = generate_problems(client, args.n_problems)
    print(f"[gen] got {len(problems)} problems")

    pairs = {"uncertainty": [], "example_generation": [], "backtracking": []}
    for i, p in enumerate(problems):
        problem = p.get("problem", "").strip()
        prefix = p.get("prefix", "").strip()
        if not problem or not prefix:
            continue
        for behaviour in pairs:
            pair = generate_pair(client, behaviour, problem, prefix)
            if pair is not None:
                pairs[behaviour].append({
                    "problem": problem,
                    "prefix": prefix,
                    "present": pair["present"].strip(),
                    "absent":  pair["absent"].strip(),
                })
        print(f"[gen] {i+1}/{len(problems)} done ({[len(v) for v in pairs.values()]})", flush=True)

    with open(args.out, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"[gen] wrote {args.out}")
    for b, ps in pairs.items():
        print(f"  {b}: {len(ps)} pairs")

if __name__ == "__main__":
    main()
