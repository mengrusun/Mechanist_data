import json, sys, os
idx = sys.argv[1]
brief_file = sys.argv[2]
base = "/data/zhenqian/hypo_eval_825/hypo_task/knowledge/efficient_reasoning"
rp = open(os.path.join(base,"scratchpad/p3/reviewer_prompt.txt")).read()
cand = json.load(open(os.path.join(base,f"scratchpad/p2/out/cand_{idx}.json")))
brief = open(brief_file).read().strip()
full = rp + "\n\n=== SEARCH SUMMARY ===\n" + brief + "\n\n=== PROPOSAL (JSON) ===\n" + json.dumps(cand, indent=2, ensure_ascii=False)
out = os.path.join(base,f"scratchpad/p3/prompt_{idx}.txt")
open(out,"w").write(full)
print(out)
print("LEN",len(full))
