import torch, json, os

BASE = '/data/zhenqian/models/Inter' + 'PLM-esm2-650m'
for layer in [24]:
    ldir = f'{BASE}/layer_{layer}'
    cfg = json.load(open(f'{ldir}/config.json'))
    print('Config:', cfg)
    for f in ['ae_normalized.pt', 'ae_unnormalized.pt']:
        p = f'{ldir}/{f}'
        st = torch.load(p, map_location='cpu', weights_only=False)
        print(f'\n== {f} ==')
        print('type:', type(st))
        if isinstance(st, dict):
            for k, v in st.items():
                if hasattr(v, 'shape'):
                    print(f'  key {k}: tensor shape={tuple(v.shape)} dtype={v.dtype}')
                else:
                    print(f'  key {k}: type={type(v).__name__} val={v}')
        else:
            print('object:', st)
