import pickle
d = pickle.load(open('/tmp/sprot_smoke.pkl', 'rb'))
print('entries:', len(d['entries']))
print('concepts:', len(d['concepts']))
print('first 5:', d['concepts'][:20])
print('example entry:', d['entries'][0], 'len(seq)=', len(d['seqs'][0]), 'nfeats=', len(d['anns'][0]))
for a, b, c in d['anns'][0][:5]:
    print(f'  {a}..{b}: {d["concepts"][c]}')
