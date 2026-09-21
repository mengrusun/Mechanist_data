import json, os, glob
data = json.load(open('claim_1.json'))
fields = ['Name','Title','Short Hypothesis','Related Work','Abstract','Experiments','Risk Factors and Limitations']
out = []
n = len(glob.glob('scratch_p3/review_*.json'))
# find contiguous 0..k
i = 0
while os.path.exists(f'scratch_p3/review_{i}.json'):
    rec = data[i]
    r = json.load(open(f'scratch_p3/review_{i}.json'))
    newrec = {k: rec[k] for k in fields}
    newrec['novelty'] = r['novelty']
    newrec['impact'] = r['impact']
    newrec['review'] = r['review']
    seen = set(); dedup = []
    for axis in ['novelty','impact','review']:
        for s in r[axis]['revision_notes']:
            if s not in seen:
                seen.add(s); dedup.append(s)
    newrec['revision_notes'] = dedup
    out.append(newrec)
    i += 1
tmp = 'claim_1_p3_part0.json.tmp'
json.dump(out, open(tmp,'w'), ensure_ascii=False, indent=4)
os.replace(tmp, 'claim_1_p3_part0.json')
print('flushed', len(out), 'records')
