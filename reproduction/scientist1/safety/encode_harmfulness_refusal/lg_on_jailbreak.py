"""Run Llama-Guard-3-8B directly on the jailbreak-attacked prompts (for Claim 5)."""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LLAMA_GUARD_PATH, load_model_and_tokenizer

JB_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/jailbreak_llama3"
RES_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/results"


@torch.no_grad()
def classify(model, tok, prompts, batch_size=4, device="cuda", max_new_tokens=20):
    preds = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        formatted = []
        for p in batch:
            msgs = [{"role": "user", "content": p}]
            s = tok.apply_chat_template(msgs, tokenize=False)
            formatted.append(s)
        enc = tok(formatted, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            in_len = enc.input_ids.shape[1]
            txt = tok.decode(gen[j, in_len:], skip_special_tokens=True)
            preds.append("unsafe" in txt.lower())
    return np.array(preds, dtype=bool)


def main():
    print("Loading Llama-Guard-3-8B ...")
    model, tok = load_model_and_tokenizer(LLAMA_GUARD_PATH)
    device = next(model.parameters()).device

    out = {}
    for atk in ["plain", "gcg", "dan", "persuasion", "jbb_role", "prefill"]:
        fp = os.path.join(JB_DIR, f"{atk}.npz")
        if not os.path.exists(fp):
            continue
        z = np.load(fp, allow_pickle=True)
        prompts = z["attack_prompts"] if "attack_prompts" in z.files else z["prompts"]
        prompts = [str(p) for p in prompts]
        t0 = time.time()
        pred = classify(model, tok, prompts, batch_size=4, device=device)
        el = time.time() - t0
        out[atk] = {
            "n": int(len(prompts)),
            "flag_rate": float(pred.mean()),
            "latency_total_sec": float(el),
            "latency_per_sample_sec": float(el / len(prompts)),
        }
        print(f"  LG {atk:12s}: n={len(prompts)} flag={pred.mean():.3f} per_sample={el/len(prompts)*1000:.1f}ms")

    with open(os.path.join(RES_DIR, "lg_on_jailbreak.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {os.path.join(RES_DIR, 'lg_on_jailbreak.json')}")


if __name__ == "__main__":
    main()
