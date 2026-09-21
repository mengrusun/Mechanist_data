import json, os, sys

idx = sys.argv[1]
raw_path = sys.argv[2]
raw = open(raw_path).read().strip()

def extract_json(s):
    # strip code fences
    if s.startswith("```"):
        s = s.split("```",2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```",1)[0]
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        # find first { to last }
        a = s.find("{"); b = s.rfind("}")
        return json.loads(s[a:b+1])

try:
    d = extract_json(raw)
    nov, imp, rev = d["novelty"], d["impact"], d["review"]
    merged = []
    for ax in (nov, imp, rev):
        for note in ax.get("revision_notes", []) or []:
            if note not in merged:
                merged.append(note)
    out = {"novelty": nov, "impact": imp, "review": rev, "revision_notes": merged}
    ok = True
except Exception as e:
    out = {
        "novelty": {"score": None, "reasoning": raw, "revision_notes": []},
        "impact": {"score": None, "reasoning": raw, "revision_notes": []},
        "review": {"score": None, "reasoning": raw, "revision_notes": []},
        "revision_notes": [],
    }
    ok = False

outdir = "scratchpad/p3/out"
os.makedirs(outdir, exist_ok=True)
final = os.path.join(outdir, f"cand_{idx}.json")
tmp = final + ".tmp"
with open(tmp, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp, final)
print("WROTE", final, "ok=", ok, "scores=",
      out["novelty"]["score"], out["impact"]["score"], out["review"]["score"])
