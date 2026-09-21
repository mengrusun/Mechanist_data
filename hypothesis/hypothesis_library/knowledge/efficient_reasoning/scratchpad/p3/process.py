import json, os, sys, re

def extract_json(text):
    text = text.strip()
    # strip code fences
    if text.startswith("```"):
        text = re.sub(r'^```[a-zA-Z]*\n', '', text)
        text = re.sub(r'\n```$', '', text.strip())
    try:
        return json.loads(text)
    except Exception:
        pass
    # find first { to matching last }
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except Exception:
                    return None
    return None

def process(idx, reply_path):
    raw = open(reply_path).read()
    data = extract_json(raw)
    outdir = "scratchpad/p3/out"
    os.makedirs(outdir, exist_ok=True)
    if data is None or not all(k in data for k in ("novelty","impact","review")):
        out = {
            "novelty": {"score": None, "reasoning": raw[:4000], "revision_notes": []},
            "impact": {"score": None, "reasoning": raw[:4000], "revision_notes": []},
            "review": {"score": None, "reasoning": raw[:4000], "revision_notes": []},
            "revision_notes": []
        }
    else:
        merged = []
        seen = set()
        for axis in ("novelty","impact","review"):
            for n in data.get(axis, {}).get("revision_notes", []) or []:
                if n not in seen:
                    seen.add(n); merged.append(n)
        out = {
            "novelty": data["novelty"],
            "impact": data["impact"],
            "review": data["review"],
            "revision_notes": merged
        }
    tmp = os.path.join(outdir, f".cand_{idx}.tmp")
    final = os.path.join(outdir, f"cand_{idx}.json")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, final)
    def sc(a): return out[a].get("score")
    print(f"cand_{idx}: novelty={sc('novelty')} impact={sc('impact')} review={sc('review')}")

if __name__ == "__main__":
    process(sys.argv[1], sys.argv[2])
