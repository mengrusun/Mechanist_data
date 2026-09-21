"""SAE loader for Gemma-Scope JumpReLU SAEs.

Gemma-Scope params.npz contains:
  W_enc: [d_model, d_features]  (2304, 16384)
  W_dec: [d_features, d_model]  (16384, 2304)
  b_enc: [d_features]
  b_dec: [d_model]              pre-encoder bias (subtracted from input)
  threshold: [d_features]       JumpReLU threshold
"""
import os
import numpy as np
import torch
import torch.nn as nn


class JumpReLUSAE(nn.Module):
    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold):
        super().__init__()
        self.d_model = W_enc.shape[0]
        self.d_features = W_enc.shape[1]
        self.register_buffer("W_enc", torch.tensor(W_enc, dtype=torch.float32))
        self.register_buffer("W_dec", torch.tensor(W_dec, dtype=torch.float32))
        self.register_buffer("b_enc", torch.tensor(b_enc, dtype=torch.float32))
        self.register_buffer("b_dec", torch.tensor(b_dec, dtype=torch.float32))
        self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))

    @torch.no_grad()
    def encode(self, x):
        """x: [..., d_model] -> [..., d_features] (post-JumpReLU activations)."""
        x = x.to(self.W_enc.dtype)
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        mask = (pre > self.threshold).to(pre.dtype)
        return pre * mask


def load_gemmascope_sae(root, layer, width="16k", l0=None, device="cuda"):
    """Load a gemma-scope SAE for a given layer.

    Args:
        root: e.g. /data/zhenqian/models/sae/gemma-scope-2b-pt-res
        layer: int
        width: '16k'
        l0: pick the average_l0_* variant; if None, pick a mid one.
    """
    layer_root = os.path.join(root, f"layer_{layer}", f"width_{width}")
    subs = [d for d in os.listdir(layer_root) if d.startswith("average_l0_")]
    l0_vals = sorted(int(d.split("_")[-1]) for d in subs)
    if l0 is None:
        target_l0 = l0_vals[len(l0_vals) // 2]  # median
    else:
        target_l0 = min(l0_vals, key=lambda v: abs(v - l0))
    path = os.path.join(layer_root, f"average_l0_{target_l0}", "params.npz")
    p = np.load(path)
    sae = JumpReLUSAE(p["W_enc"], p["W_dec"], p["b_enc"], p["b_dec"], p["threshold"]).to(device)
    sae.eval()
    return sae, {"layer": layer, "width": width, "l0": target_l0, "path": path}


if __name__ == "__main__":
    import sys
    sae, info = load_gemmascope_sae(
        "/data/zhenqian/models/sae/gemma-scope-2b-pt-res", layer=12, width="16k", l0=82
    )
    print(info)
    print("d_model=", sae.d_model, "d_features=", sae.d_features)
    x = torch.randn(2, 5, sae.d_model, device="cuda") * 3
    z = sae.encode(x)
    print("z shape:", z.shape, "nonzero fraction:", (z > 0).float().mean().item())
