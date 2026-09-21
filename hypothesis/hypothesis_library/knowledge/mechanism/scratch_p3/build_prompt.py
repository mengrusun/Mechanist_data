import json, sys
idx = int(sys.argv[1])
brief_path = sys.argv[2]
rev = open('scratch_p3/reviewer_prompt.txt').read()
brief = open(brief_path).read().strip()
rec = json.load(open(f'scratch_p3/rec_{idx}.json'))
prompt = rev + "\nSEARCH SUMMARY:\n" + brief + "\nPROPOSAL (JSON):\n" + json.dumps(rec, ensure_ascii=False)
open(f'scratch_p3/prompt_{idx}.txt','w').write(prompt)
print(f"prompt_{idx}.txt bytes:", len(prompt))
