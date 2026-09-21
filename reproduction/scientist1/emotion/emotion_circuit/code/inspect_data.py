import json, sys

paths = [
    '/data/zhenqian/Reproduction1/cc/emotion/emotion_circuit/data/sev.jsonl',
    '/data/zhenqian/Reproduction1/cc/emotion/emotion_circuit/data/test_set.jsonl',
]
for p in paths:
    with open(p) as f:
        rows = [json.loads(l) for l in f]
    themes = {}
    for r in rows:
        themes[r['theme']] = themes.get(r['theme'], 0) + 1
    print(p)
    print('  rows=', len(rows), 'events=', len(rows) * 3)
    print('  themes:', themes)
    print('  example keys:', list(rows[0].keys()))
    print('  example scenario:', rows[0]['scenario'])
