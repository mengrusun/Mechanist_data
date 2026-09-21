"""Qwen3-4B transcoder SAE loader.

The transcoder maps mlp_hook_in -> mlp_hook_out via a sparse feature space.
Weights are stored as `layer_L.safetensors`; we inspect keys at runtime.

Common conventions:
  W_enc: [d_model, d_features]
  W_dec: [d_features, d_model]
  b_enc: [d_features]
  b_dec: [d_model]     (subtracted from input before encoding)
  activation: 'relu' or 'jumprelu' or 'topk'
"""
import os
import torch
import torch.nn as nn
from safetensors.torch import load_file


class Transcoder(nn.Module):
    """Qwen3-4B transcoder. Weights stored as:
        W_enc: [d_features, d_model_in]  (rows are encoder directions)
        W_dec: [d_features, d_model_out] (rows are decoder directions)
        b_enc: [d_features]
        b_dec: [d_model_out]
    Only encode() is used for our SAGE pipeline.
    """
    def __init__(self, W_enc, W_dec, b_enc, b_dec, activation="relu", threshold=None):
        super().__init__()
        # standardize to [d_features, d_model_in]
        assert W_enc.dim() == 2
        self.d_features, self.d_model_in = W_enc.shape
        self.d_model_out = W_dec.shape[1]
        self.register_buffer("W_enc", W_enc)  # [F, D]
        self.register_buffer("W_dec", W_dec)  # [F, D']
        self.register_buffer("b_enc", b_enc)
        self.register_buffer("b_dec", b_dec)
        self.activation = activation
        if threshold is not None:
            self.register_buffer("threshold", threshold)
        else:
            self.threshold = None

    @torch.no_grad()
    def encode(self, x):
        """x: [..., d_model_in] -> [..., d_features]."""
        x = x.to(self.W_enc.dtype)
        # W_enc is [F, D], so we compute x @ W_enc.T -> [..., F]
        pre = torch.einsum("...d,fd->...f", x, self.W_enc) + self.b_enc
        if self.activation == "relu":
            return torch.relu(pre).float()
        elif self.activation == "jumprelu":
            return (pre * (pre > self.threshold).to(pre.dtype)).float()
        else:
            return torch.relu(pre).float()


def load_qwen_transcoder(path, device="cuda"):
    """Load one layer's transcoder safetensors."""
    tensors = load_file(path)
    W_enc = tensors["W_enc"]
    W_dec = tensors["W_dec"]
    b_enc = tensors["b_enc"]
    b_dec = tensors["b_dec"]
    thresh = tensors.get("threshold")
    activation = "jumprelu" if thresh is not None else "relu"
    tc = Transcoder(W_enc.to(device), W_dec.to(device), b_enc.to(device), b_dec.to(device),
                    activation=activation,
                    threshold=(thresh.to(device) if thresh is not None else None))
    tc.eval()
    return tc


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "/data/zhenqian/models/sae/qwen3-4b-transcoders-weights/layer_12.safetensors"
    tc = load_qwen_transcoder(p)
    print("Loaded transcoder d_features=", tc.d_features)
    x = torch.randn(2, 5, tc.d_model_in, device="cuda")
    z = tc.encode(x)
    print("z shape:", z.shape, "nonzero frac:", (z > 0).float().mean().item())
