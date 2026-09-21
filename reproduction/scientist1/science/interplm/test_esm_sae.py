"""Sanity: run ESM-2-650M on a toy sequence, hook layer 24 residual stream,
pass through SAE, check shapes and reconstruction quality."""
import torch
from transformers import AutoTokenizer, AutoModel
from sae_module import load_sae

DEVICE = 'cuda:0'
ESM_DIR = '/data/zhenqian/models/esm2_t33_650M_UR50D'
SAE_DIR = '/data/zhenqian/models/Inter' + 'PLM-esm2-650m/layer_24'
LAYER = 24  # residual-stream after transformer block index 24 (0-indexed layer output)

print('loading tokenizer and model...')
tok = AutoTokenizer.from_pretrained(ESM_DIR)
model = AutoModel.from_pretrained(ESM_DIR, torch_dtype=torch.float32).to(DEVICE).eval()
print('num layers:', model.config.num_hidden_layers)

sae = load_sae(SAE_DIR, normalized=True, device=DEVICE)

seq = 'MKTIIALSYIFCLVFADYKDDDDKAAAAAAAAAAAAAAAAAAAAAA'
enc = tok(seq, return_tensors='pt').to(DEVICE)
with torch.no_grad():
    out = model(**enc, output_hidden_states=True)
# hidden_states is a tuple of len (num_layers+1) - includes embeddings
# hidden_states[k] = output of layer k (with embeddings at index 0)
print('num hidden states:', len(out.hidden_states))
h = out.hidden_states[LAYER]  # shape (1, T, 1280)
print('h shape:', h.shape, 'dtype:', h.dtype)

with torch.no_grad():
    x_hat, z = sae(h)
print('z (features) shape:', z.shape, 'nnz per token:', (z > 0).float().mean().item())
mse = ((x_hat - h)**2).mean().item()
h_mag = (h**2).mean().item()
print(f'recon MSE={mse:.4f}, orig mag={h_mag:.4f}, frac var explained ~ {1 - mse/h_mag:.3f}')

# also unnormalized
sae2 = load_sae(SAE_DIR, normalized=False, device=DEVICE)
with torch.no_grad():
    x_hat2, z2 = sae2(h)
mse2 = ((x_hat2 - h)**2).mean().item()
print(f'unnormalized: recon MSE={mse2:.4f}, nnz/tok={(z2>0).float().mean().item():.3f}')
