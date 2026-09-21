import json, sys
idx = sys.argv[1]
rp = open("reviewer_prompt.txt").read()
brief = open(f"brief_{idx}.txt").read().strip()
cand = json.load(open(f"../p2/out/cand_{idx}.json"))
pretty = json.dumps(cand, indent=2, ensure_ascii=False)
full = rp + "\n\n=== SEARCH SUMMARY ===\n" + brief + "\n\n=== PROPOSAL (JSON) ===\n" + pretty + "\n\nReturn strict JSON only, no prose around it."
open(f"prompt_{idx}.txt","w").write(full)
print(f"prompt_{idx}.txt built, {len(full)} chars")
