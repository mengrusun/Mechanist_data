import importlib
mods = ['torch', 'transformers', 'pandas', 'numpy', 'tqdm', 'sklearn', 'safetensors', 'einops', 'Bio']
for m in mods:
    try:
        importlib.import_module(m)
        print(f'OK  {m}')
    except Exception as e:
        print(f'ERR {m}: {e}')
import torch
print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())
