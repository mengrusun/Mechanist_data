import json, sys, os, re
idx = sys.argv[1]
reply_file = sys.argv[2]
base = "/data/zhenqian/hypo_eval_825/hypo_task/knowledge/efficient_reasoning"
raw = open(reply_file).read()
def extract_json(s):
    s=s.strip()
    if s.startswith("```"):
        s=re.sub(r'^```[a-zA-Z]*\n','',s); s=re.sub(r'\n```$','',s.strip())
    try:
        return json.loads(s)
    except Exception:
        m=re.search(r'\{.*\}', s, re.DOTALL)
        if m: return json.loads(m.group(0))
        raise
try:
    d=extract_json(raw)
    nov,imp,rev=d['novelty'],d['impact'],d['review']
    def norm(x):
        sc=x.get('score')
        sc=int(sc) if isinstance(sc,(int,float)) or (isinstance(sc,str) and sc.strip().isdigit()) else None
        return {"score":sc,"reasoning":x.get('reasoning',''),"revision_notes":list(x.get('revision_notes') or [])}
    nov,imp,rev=norm(nov),norm(imp),norm(rev)
    merged=[]; seen=set()
    for lst in (nov['revision_notes'],imp['revision_notes'],rev['revision_notes']):
        for n in lst:
            if n not in seen:
                seen.add(n); merged.append(n)
    out={"novelty":nov,"impact":imp,"review":rev,"revision_notes":merged}
except Exception as e:
    out={"novelty":{"score":None,"reasoning":"PARSE_FAIL: "+str(e)+" RAW: "+raw,"revision_notes":[]},
         "impact":{"score":None,"reasoning":"PARSE_FAIL","revision_notes":[]},
         "review":{"score":None,"reasoning":"PARSE_FAIL","revision_notes":[]},
         "revision_notes":[]}
path=os.path.join(base,f"scratchpad/p3/out/cand_{idx}.json")
tmp=path+".tmp"
with open(tmp,"w") as f:
    json.dump(out,f,indent=2,ensure_ascii=False); f.flush(); os.fsync(f.fileno())
os.replace(tmp,path)
print(f"WROTE cand_{idx}: novelty={out['novelty']['score']} impact={out['impact']['score']} review={out['review']['score']}")
