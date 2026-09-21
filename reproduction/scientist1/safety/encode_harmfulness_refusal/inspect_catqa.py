import json
lines = []
with open('/data/zhenqian/data/CATQA/data/catqa_english.json') as f:
    for i, line in enumerate(f):
        lines.append(line.strip())
        if i >= 3: break
for l in lines:
    print(l[:300])
