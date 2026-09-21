"""Extract last-token hidden states across all layers for a parallel multilingual probe set.

For each language, we tokenize the same MGSM problem (test split) formatted as a chat prompt
(with add_generation_prompt) and record the hidden state at the last position at every layer,
including the embedding layer. Hidden states are saved as float16 to save disk.

Output: cache/hidden_states.npz  (dict-like with h_{lang}: [N_probe, L+1, D])
        cache/probe_meta.json    (list of probe indices used)
"""
import os, sys, time, json, gc
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from common import MODEL_PATH, LANGS, load_mgsm, build_prompt, apply_chat

N_PROBE = 64          # number of parallel test problems used as probes
BATCH = 8             # micro-batch through the encoder-only forward
OUT_DIR = "cache"
os.makedirs(OUT_DIR, exist_ok=True)

# For probe hidden states, do not enable thinking (we want the last-token rep of the *prompt*)
ENABLE_THINK = False


def main():
    print("loading tokenizer / model")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    print(f"layers={n_layers} hidden={d}")

    print("loading MGSM test splits")
    data = load_mgsm("test")

    n_test = len(data["en"])
    probe_idx = list(range(min(N_PROBE, n_test)))
    # save meta
    with open(os.path.join(OUT_DIR, "probe_meta.json"), "w") as f:
        json.dump({"indices": probe_idx, "N_PROBE": len(probe_idx)}, f)

    # (n_probe, n_layers+1, d) per language
    hs_per_lang = {}
    for lg in LANGS:
        print(f"[{lg}] extracting hidden states, {len(probe_idx)} probes")
        prompts = []
        for i in probe_idx:
            q = data[lg].iloc[i]["question"]
            prompts.append(apply_chat(tok, build_prompt(q, lg), enable_thinking=ENABLE_THINK))
        H = np.zeros((len(probe_idx), n_layers + 1, d), dtype=np.float16)
        with torch.inference_mode():
            for b0 in range(0, len(prompts), BATCH):
                b_prompts = prompts[b0:b0 + BATCH]
                enc = tok(b_prompts, return_tensors="pt", padding=True, truncation=True,
                          max_length=1024).to("cuda:0")
                out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
                # hidden_states is a tuple of length n_layers+1 (embed + each layer output)
                # each of shape (B, T, D). We want the *last non-pad* token per sample.
                attn = enc["attention_mask"]              # (B, T)
                last_idx = attn.sum(dim=1) - 1            # (B,)
                for j in range(len(b_prompts)):
                    idx = int(last_idx[j].item())
                    for k, h in enumerate(out.hidden_states):
                        H[b0 + j, k] = h[j, idx].to(torch.float16).cpu().numpy()
                del out, enc
                if b0 % (BATCH * 4) == 0:
                    torch.cuda.empty_cache()
        hs_per_lang[lg] = H
        print(f"  saved shape {H.shape}, dtype {H.dtype}")

    out_path = os.path.join(OUT_DIR, "hidden_states.npz")
    np.savez_compressed(out_path, **{f"h_{k}": v for k, v in hs_per_lang.items()})
    print("wrote", out_path)


if __name__ == "__main__":
    main()
