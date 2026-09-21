"""Qwen3-4B hook: capture MLP input (= post_attention_layernorm output) at layer L.

Transcoder maps mlp.hook_in -> mlp.hook_out. We feed mlp.hook_in to the transcoder.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenHooked:
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device
        )
        self.model.eval()
        self.device = device
        self.dtype = dtype
        self._captured = None
        self._h = None

    def _mlp_pre_hook(self, module, inputs):
        # inputs is a tuple; first arg is post-layernorm output (MLP input)
        x = inputs[0]
        self._captured = x.detach()

    def register_hook(self, layer):
        self.clear_hook()
        target = self.model.model.layers[layer].mlp
        self._h = target.register_forward_pre_hook(self._mlp_pre_hook)

    def clear_hook(self):
        if self._h is not None:
            self._h.remove()
            self._h = None
        self._captured = None

    @torch.no_grad()
    def get_mlp_in(self, texts, layer, max_length=128):
        self.register_hook(layer)
        try:
            enc = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=max_length, add_special_tokens=True,
            ).to(self.device)
            _ = self.model(**enc)
            return self._captured, enc.input_ids, enc.attention_mask
        finally:
            self.clear_hook()

    @torch.no_grad()
    def feature_activation_of(self, texts, transcoder, layer, feature_idx, max_length=128, mode="max"):
        resid, input_ids, attn = self.get_mlp_in(texts, layer, max_length=max_length)
        # Encode this feature only, to avoid materializing [B, T, 163840]
        # Take one row of W_enc for the feature
        W_row = transcoder.W_enc[feature_idx]  # [D]
        b = transcoder.b_enc[feature_idx].float()
        r = resid.to(transcoder.W_enc.dtype)
        pre = torch.einsum("btd,d->bt", r, W_row).float() + b
        z = torch.relu(pre)  # [B, T]

        attn_f = attn.to(z.dtype)
        z_masked = z * attn_f + (attn_f - 1) * 1e9  # -inf on padded
        vals, idxs = z_masked.max(dim=1)
        return vals.cpu().tolist(), idxs.cpu().tolist(), {"input_ids": input_ids, "attention_mask": attn}


if __name__ == "__main__":
    from qwen_sae import load_qwen_transcoder
    qm = QwenHooked("/data/zhenqian/models/Qwen3-4B")
    tc = load_qwen_transcoder("/data/zhenqian/models/sae/qwen3-4b-transcoders-weights/layer_12.safetensors")
    texts = [
        "def hello():\n    print('hi')",
        "The capital of France is Paris.",
        "Learning rate warmup helps stabilize training.",
    ]
    for f in [0, 100, 500, 1000]:
        vals, idxs, out = qm.feature_activation_of(texts, tc, layer=12, feature_idx=f)
        print(f"feature {f} vals:", [f"{v:.2f}" for v in vals])
