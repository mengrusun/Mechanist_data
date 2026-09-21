"""Cache residual-stream activations at every layer of Qwen-3-4B-Thinking.

Given (model, tokenizer, list of texts, list of layer indices), returns:
    a dict `layer_idx -> np.ndarray of shape (N, hidden_size)`
where each row is the activation at the last non-pad token of the input.

For memory reasons, we cache one language / probe-set at a time and store the results as .npz.
"""

import gc
from typing import Dict, List, Optional

import numpy as np
import torch


class ResidualStreamCache:
    def __init__(self, model, tokenizer, layer_indices: List[int], pool: str = "last"):
        """
        Args:
            model: the HF causal LM (Qwen3ForCausalLM).
            tokenizer: the corresponding tokenizer (padding_side="left" recommended).
            layer_indices: list of transformer block indices to cache (0-based, up to num_hidden_layers - 1).
            pool: "last" (last non-pad token) or "mean" (mean over non-pad tokens).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layer_indices = sorted(set(int(x) for x in layer_indices))
        self.pool = pool
        self.hooks = []
        self._cur_activations: Dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            # For a Qwen3DecoderLayer, the output is a tuple whose first element is the residual stream tensor
            hidden = output[0] if isinstance(output, tuple) else output
            self._cur_activations[layer_idx] = hidden.detach()
            return output
        return hook_fn

    def __enter__(self):
        layers = self.model.model.layers
        for idx in self.layer_indices:
            h = layers[idx].register_forward_hook(self._make_hook(idx))
            self.hooks.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self._cur_activations = {}

    def encode_batch(self, texts: List[str], max_length: int = 256) -> Dict[int, np.ndarray]:
        """Run one forward pass on a batch and return {layer_idx: (batch, hidden)} pooled arrays."""
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(self.model.device)
        attn = enc["attention_mask"].to(self.model.device)
        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        # Pool
        pooled_out: Dict[int, np.ndarray] = {}
        for idx, hidden in self._cur_activations.items():
            # hidden: (batch, seq, hidden)
            if self.pool == "last":
                # last non-pad token index per row
                # left padding: last real token is at position seq-1 for every row
                # right padding: last real token = sum(attn) - 1
                if self.tokenizer.padding_side == "left":
                    pooled = hidden[:, -1, :]
                else:
                    lens = attn.sum(dim=1) - 1
                    pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lens]
            elif self.pool == "mean":
                mask = attn.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            else:
                raise ValueError(f"Unknown pool: {self.pool}")
            pooled_out[idx] = pooled.to(torch.float32).cpu().numpy()
        self._cur_activations.clear()
        return pooled_out

    def encode_dataset(
        self,
        texts: List[str],
        batch_size: int = 16,
        max_length: int = 256,
    ) -> Dict[int, np.ndarray]:
        """Encode a full list of texts, returning stacked (N, hidden) arrays per layer."""
        per_layer: Dict[int, List[np.ndarray]] = {idx: [] for idx in self.layer_indices}
        n = len(texts)
        for i in range(0, n, batch_size):
            batch = texts[i : i + batch_size]
            pooled = self.encode_batch(batch, max_length=max_length)
            for idx, arr in pooled.items():
                per_layer[idx].append(arr)
        out = {idx: np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0)) for idx, chunks in per_layer.items()}
        return out
