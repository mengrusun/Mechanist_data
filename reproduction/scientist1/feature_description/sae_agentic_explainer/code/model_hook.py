"""Model wrapper: run text through target LLM, capture residual stream at layer L, compute SAE activations."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HookedModel:
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device
        )
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self._captured = None
        self._hook_handle = None

    def _resid_module(self, layer):
        # Gemma-2: model.layers[layer] is a decoder block. Residual stream after the block
        # is captured by hooking its forward hook, which returns hidden_states.
        return self.model.model.layers[layer]

    def _hook(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        self._captured = h.detach()

    def register_hook(self, layer):
        self.clear_hook()
        h = self._resid_module(layer)
        self._hook_handle = h.register_forward_hook(self._hook)

    def clear_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._captured = None

    @torch.no_grad()
    def get_residual(self, texts, layer, max_length=128):
        """Return residual stream at `layer` after decoder block. Shape [B, T, d_model]."""
        self.register_hook(layer)
        try:
            enc = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=max_length, add_special_tokens=True,
            ).to(self.device)
            _ = self.model(**enc)
            resid = self._captured
            return resid, enc.input_ids, enc.attention_mask
        finally:
            self.clear_hook()

    @torch.no_grad()
    def feature_activations(self, texts, sae, layer, max_length=128):
        """Return per-token SAE activations for a given feature-agnostic pass.

        Returns dict:
          input_ids: [B, T]
          attention_mask: [B, T]
          resid: [B, T, d_model] (fp32 for numerical stability of SAE math)
          z: [B, T, d_features]  (only if features_of_interest is None; may be huge!)

        To avoid materializing [B,T,d_features] for all 16k features when we only care about a few,
        callers can pass features_of_interest.
        """
        resid, input_ids, attn = self.get_residual(texts, layer, max_length=max_length)
        resid_f = resid.float()
        z = sae.encode(resid_f)  # [B, T, d_features]
        return {"input_ids": input_ids, "attention_mask": attn, "resid": resid_f, "z": z}

    @torch.no_grad()
    def feature_activation_of(self, texts, sae, layer, feature_idx, max_length=128, mode="max"):
        """Return the per-text scalar activation of a single feature.

        mode: 'max' (over tokens) or 'mean' (over unmasked tokens).
        Also returns the argmax token (index within input_ids) for interpretability.
        """
        out = self.feature_activations(texts, sae, layer, max_length=max_length)
        z_feat = out["z"][..., feature_idx]  # [B, T]
        attn = out["attention_mask"].to(z_feat.dtype)
        z_masked = z_feat * attn + (attn - 1) * 1e9  # -inf on padded
        if mode == "max":
            vals, idxs = z_masked.max(dim=1)
            return vals.cpu().tolist(), idxs.cpu().tolist(), out
        elif mode == "mean":
            denom = attn.sum(dim=1).clamp_min(1)
            vals = (z_feat * attn).sum(dim=1) / denom
            return vals.cpu().tolist(), None, out
        else:
            raise ValueError(mode)


if __name__ == "__main__":
    from sae import load_gemmascope_sae
    hm = HookedModel("/data/zhenqian/models/gemma-2-2b")
    sae, _ = load_gemmascope_sae(
        "/data/zhenqian/models/sae/gemma-scope-2b-pt-res", layer=12, width="16k", l0=82
    )
    texts = ["The capital of France is Paris.", "Cats are wonderful pets."]
    vals, idxs, _ = hm.feature_activation_of(texts, sae, layer=12, feature_idx=0)
    print("feature 0 vals:", vals, "argmax_tok:", idxs)
