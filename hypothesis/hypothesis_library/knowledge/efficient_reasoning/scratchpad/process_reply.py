import json, sys, os, re
BASE='/data/zhenqian/hypo_eval_825/hypo_task/knowledge/efficient_reasoning'
idx = sys.argv[1]
reply_path = sys.argv[2]
raw = open(reply_path).read()
def extract_json(s):
    s=s.strip()
    m=re.search(r'\{.*\}', s, re.DOTALL)
    if m: s=m.group(0)
    return json.loads(s)
try:
    obj = extract_json(raw)
    def norm(a):
        return {'score': (int(a['score']) if a.get('score') is not None else None),
                'reasoning': a.get('reasoning',''),
                'revision_notes': list(a.get('revision_notes',[]))}
    nov=norm(obj['novelty']); imp=norm(obj['impact']); rev=norm(obj['review'])
    merged=[]; seen=set()
    for a in (nov,imp,rev):
        for n in a['revision_notes']:
            if n not in seen: seen.add(n); merged.append(n)
    out={'novelty':nov,'impact':imp,'review':rev,'revision_notes':merged}
except Exception as e:
    out={'novelty':{'score':None,'reasoning':raw,'revision_notes':[]},
         'impact':{'score':None,'reasoning':raw,'revision_notes':[]},
         'review':{'score':None,'reasoning':raw,'revision_notes':[]},'revision_notes':[]}
    print('PARSE_FAIL', e, file=sys.stderr)
final=os.path.join(BASE,'scratchpad/p3/out', f'cand_{idx}.json')
tmp=final+'.tmp'
with open(tmp,'w') as f:
    json.dump(out,f,indent=2,ensure_ascii=False); f.flush(); os.fsync(f.fileno())
os.replace(tmp, final)
print('WROTE cand_'+idx, 'scores n/i/r:', out['novelty']['score'], out['impact']['score'], out['review']['score'])
