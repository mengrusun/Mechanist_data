"""Standalone SAE implementation matching the shipped checkpoint layout.

State-dict keys in ae_normalized.pt / ae_unnormalized.pt:
    bias:            (d_in,)      pre-encoder / decoder bias (b_dec)
    encoder.weight:  (d_sae, d_in)
    encoder.bias:    (d_sae,)
    decoder.weight:  (d_in, d_sae)
Forward:
    z = ReLU(encoder(x - bias) + encoder.bias)
    x_hat = decoder(z) + bias
"""
from __future__ import annotations
import json
import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_in))  # b_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x - self.bias))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.bias

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    @classmethod
    def from_pretrained(cls, layer_dir: str, normalized: bool = True) -> "SparseAutoencoder":
        cfg = json.load(open(f"{layer_dir}/config.json"))
        arch = cfg["architecture"]
        model = cls(arch["esm_dim"], arch["feature_dim"])
        name = "ae_normalized.pt" if normalized else "ae_unnormalized.pt"
        sd = torch.load(f"{layer_dir}/{name}", map_location="cpu", weights_only=False)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model


def load_sae(layer_dir: str, normalized: bool = True, device: str = "cuda") -> SparseAutoencoder:
    sae = SparseAutoencoder.from_pretrained(layer_dir, normalized=normalized)
    return sae.to(device)
