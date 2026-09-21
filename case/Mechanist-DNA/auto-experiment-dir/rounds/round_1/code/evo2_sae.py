"""
Shared Evo2-7B + Layer-26 BatchTopK SAE utilities.

HC1: uses ONLY Evo2-7B (/data1/share_model/evo2/evo2_7b/evo2_7b.pt) +
the pre-trained Layer-26 mixed BatchTopK SAE
(/data1/share_model/evo2/evo2_sae_layer26_mixed/sae-layer26-mixed-expansion_8-k_64.pt).

SAE format (verified from checkpoint):
  _orig_mod.W      (4096, 32768)   encoder weight (tied: decoder = W^T)
  _orig_mod.b_enc  (32768,)        encoder bias
  _orig_mod.b_dec  (4096,)         decoder bias (pre-subtracted from input)

Encode:  z_pre = (x - b_dec) @ W + b_enc ;  z = topk(z_pre, k=64) with ReLU
Decode:  x_hat = z @ W^T + b_dec

The Evo2 residual-stream hook site for the Layer-26 SAE is the OUTPUT of
`blocks.26` (the post-block residual stream), captured with Evo2.forward(
return_embeddings=True, layer_names=[EVO2_SAE_LAYER]).
"""
import os
import torch
import torch.nn.functional as F

EVO2_7B_PATH = "/data1/share_model/evo2/evo2_7b/evo2_7b.pt"
SAE_PATH = "/data1/share_model/evo2/evo2_sae_layer26_mixed/sae-layer26-mixed-expansion_8-k_64.pt"
EVO2_SAE_LAYER = "blocks.26.mlp.l3"  # overridden after empirical probe; see resolve_sae_layer
SAE_K = 64
SAE_DICT = 32768
HIDDEN = 4096


def load_evo2(device: str = "cuda:0"):
    """Load Evo2-7B via vortex. Device placement handled by vortex/CUDA_VISIBLE_DEVICES."""
    from evo2 import Evo2
    model = Evo2(model_name="evo2_7b", local_path=EVO2_7B_PATH)
    return model


class BatchTopKSAE:
    """Tied-weight BatchTopK SAE (inference: per-token top-k=64, ReLU-gated)."""

    def __init__(self, path: str = SAE_PATH, device: str = "cuda:0", dtype=torch.float32,
                 normalize: str = "none"):
        sd = torch.load(path, map_location="cpu")
        # strip _orig_mod. prefix
        W = sd["_orig_mod.W"].to(device=device, dtype=dtype)        # (4096, 32768)
        b_enc = sd["_orig_mod.b_enc"].to(device=device, dtype=dtype)  # (32768,)
        b_dec = sd["_orig_mod.b_dec"].to(device=device, dtype=dtype)  # (4096,)
        assert W.shape == (HIDDEN, SAE_DICT), f"unexpected W shape {W.shape}"
        self.W = W
        self.b_enc = b_enc
        self.b_dec = b_dec
        self.device = device
        self.dtype = dtype
        self.k = SAE_K
        # input normalization applied before encode (Goodfire/Anthropic SAE convention):
        #   'unit_sqrtd' -> per-token L2-normalize then scale by sqrt(d); None -> raw; 'scale:<c>'
        self.normalize = normalize

    def _norm(self, x):
        if self.normalize is None or self.normalize == "none":
            return x
        if self.normalize == "unit_sqrtd":
            import math
            return x / (x.norm(dim=-1, keepdim=True) + 1e-8) * math.sqrt(HIDDEN)
        if isinstance(self.normalize, str) and self.normalize.startswith("scale:"):
            c = float(self.normalize.split(":")[1])
            return x / c
        return x

    def encode_pre(self, x):
        """Pre-activation (no top-k), shape (..., 32768)."""
        x = self._norm(x.to(self.device, self.dtype))
        return (x - self.b_dec) @ self.W + self.b_enc

    def encode(self, x, k: int = None):
        """Top-k=64, ReLU-gated feature activations, shape (..., 32768)."""
        k = k or self.k
        z_pre = self.encode_pre(x)
        z_relu = F.relu(z_pre)
        # per-token top-k
        vals, idx = torch.topk(z_relu, k, dim=-1)
        z = torch.zeros_like(z_relu)
        z.scatter_(-1, idx, vals)
        return z

    def decode(self, z):
        return z @ self.W.t() + self.b_dec

    def roundtrip(self, x):
        z = self.encode(x)
        return self.decode(z), z


def resolve_sae_layer(model, seq_len_probe: int = 128):
    """Return the submodule name whose output is a (B, L, 4096) residual tensor at block 26.
    Tries a small set of candidate names; returns the first that yields hidden==4096."""
    candidates = [
        "blocks.26.mlp.l3",
        "blocks.26",
        "blocks.26.pre_norm",
        "blocks.26.post_norm",
    ]
    tok = torch.tensor([model.tokenizer.tokenize("ACGT" * (seq_len_probe // 4))],
                       dtype=torch.long, device="cuda:0")
    valid = []
    for name in candidates:
        try:
            _, emb = model(tok, return_embeddings=True, layer_names=[name])
            t = emb[name]
            if t.shape[-1] == HIDDEN:
                valid.append((name, tuple(t.shape)))
        except Exception as e:
            pass
    return valid


def tokenize_seq(model, seq: str, device: str = "cuda:0"):
    ids = model.tokenizer.tokenize(seq)
    return torch.tensor([ids], dtype=torch.long, device=device)
