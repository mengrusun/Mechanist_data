"""
Generate a small (default 15) eval problem set for the steering experiments.
These problems must NOT overlap with the contrastive-pair problems.
"""
import os, json, argparse, sys
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
    os.environ.pop(k, None)
os.environ["no_proxy"] = "*"

from openai import OpenAI

API_KEY  = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL    = "gpt-5.4"

PROMPT = """Generate {n} short, self-contained reasoning problems. Each should be
solvable with a chain of thought and have a unique correct short answer (number,
fraction, or short phrase). Aim for medium difficulty — harder than trivial 1+1
but not competition-hard. Cover a mix of arithmetic word problems, algebra,
geometry, combinatorics, logic, probability.

Return STRICTLY a JSON list of {n} objects, no code fence. Each object has:
  "problem": <str>,
  "answer":  <str  (short reference answer)>,
  "category": <str>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", default="data/eval_problems.json")
    args = ap.parse_args()

    c = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    r = c.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content": PROMPT.format(n=args.n)}],
        max_tokens=6000,
    )
    text = r.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # find first [ ... ]
    s = text.find("["); e = text.rfind("]")
    probs = json.loads(text[s:e+1])
    with open(args.out, "w") as f:
        json.dump(probs, f, indent=2)
    print(f"[save] wrote {len(probs)} problems -> {args.out}")

if __name__ == "__main__":
    main()
