"""Send each feature's top-activating protein contexts to an LLM and ask what
biological concept the feature might represent, plus a self-scored confidence
between 0 and 1.

Reads the top-contexts pickle produced by find_top_contexts.py.
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import time
import urllib.request
import urllib.error

API_URL = "https://www.dmxapi.cn/v1/chat/completions"
API_KEY = "<Your_api>"
MODEL = "gpt-5.4"

PROMPT = """You are a biologist reading a sparse-autoencoder feature that fires
on residues of protein sequences. Below are the top-activating windows around
the peak residue (the peak is in [brackets]).

{contexts}

Task: Propose ONE biological concept this feature most likely detects. Focus
on residue-level biology (motif, active/binding site, structural element,
PTM site, glycosylation, coiled-coil, transmembrane, signal peptide, disulfide,
metal binding, etc.).

Reply strictly as one line of JSON:
{{"label": "<short concept>", "reason": "<one sentence>", "confidence": <0..1>}}"""


def build_prompt(contexts):
    lines = []
    for i, (val, pi, pos, mark_seq, ac) in enumerate(contexts[:15], 1):
        lines.append(f"{i}. act={val:.2f}  {ac}@{pos}  {mark_seq}")
    return PROMPT.format(contexts="\n".join(lines))


def call_api(prompt: str, retries: int = 3, timeout: int = 90) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    proxy_handler = urllib.request.ProxyHandler({})  # bypass proxy per note
    opener = urllib.request.build_opener(proxy_handler)
    for k in range(retries):
        try:
            with opener.open(req, timeout=timeout) as f:
                data = json.load(f)
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body_bytes = e.read()
            print(f"HTTPError {e.code}: {body_bytes[:200]!r}")
            time.sleep(2 ** k)
        except Exception as e:
            print(f"[retry {k+1}] {e}")
            time.sleep(2 ** k)
    raise RuntimeError("API failed")


def parse_json(txt: str) -> dict:
    # LLM may wrap in ```json ... ``` or add prose
    import re
    m = re.search(r"\{[^{}]*\}", txt)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"label": txt.strip()[:80], "reason": "", "confidence": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    with open(args.contexts, "rb") as f:
        ctx = pickle.load(f)

    out = {}
    ids = list(ctx.keys())
    if args.limit:
        ids = ids[:args.limit]
    for i, fid in enumerate(ids):
        if not ctx[fid]:
            continue
        prompt = build_prompt(ctx[fid])
        try:
            resp = call_api(prompt)
            parsed = parse_json(resp)
        except Exception as e:
            parsed = {"label": None, "reason": str(e), "confidence": None}
        parsed["feature"] = fid
        out[fid] = parsed
        print(f"[{i+1}/{len(ids)}] f={fid}  ->  {parsed.get('label')} (conf={parsed.get('confidence')})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
