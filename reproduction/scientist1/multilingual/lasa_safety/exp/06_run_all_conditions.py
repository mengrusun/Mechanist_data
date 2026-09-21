"""
Load LLaMA-3.1-8B-Instruct once and run generation on MultiJail under a list of
(layer, alpha) conditions.

Conditions with layer=-1 disable steering (baseline).
"""

import os, argparse, json, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_chat_ids(tokenizer, user_text, device):
    msgs = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(device)


class ResidualSteer:
    def __init__(self, model, layer, v, alpha):
        self.model = model
        self.layer = layer
        self.v = v
        self.alpha = alpha
        self.handle = None
    def __enter__(self):
        mod = self.model.model.layers[self.layer]
        v_local = self.v.to(next(mod.parameters()).dtype)
        alpha = self.alpha
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
                h = h + alpha * v_local.view(1, 1, -1)
                return (h,) + output[1:]
            else:
                return output + alpha * v_local.view(1, 1, -1)
        self.handle = mod.register_forward_hook(hook)
        return self
    def __exit__(self, *a):
        if self.handle is not None: self.handle.remove()

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhenqian/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--multijail", default="/data/zhenqian/data/multijail/MultiJail.csv")
    ap.add_argument("--refusal_dir", default="cache/refusal_dir.npz")
    ap.add_argument("--n_prompts", type=int, default=40)
    ap.add_argument("--langs", nargs="+",
                    default=["en","zh","it","vi","ar","ko","th","bn","sw","jv"])
    ap.add_argument("--conditions", type=str, default="",
                    help='Comma-separated "name:layer:alpha", e.g., "baseline:-1:0,bottleneck:14:5,surface:28:5"')
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()

    d = np.load(args.refusal_dir)
    v_unit = d["v_unit"]  # [L+1, H]

    conditions = []
    for c in args.conditions.split(","):
        if not c.strip(): continue
        name, layer, alpha = c.split(":")
        conditions.append({"name": name, "layer": int(layer), "alpha": float(alpha)})

    df = pd.read_csv(args.multijail).head(args.n_prompts).reset_index(drop=True)
    prompts_by_lang = {lang: df[lang].tolist() for lang in args.langs}

    for cond in conditions:
        out_path = os.path.join(args.out_dir, f"gen_{cond['name']}_L{cond['layer']}_a{cond['alpha']}.jsonl")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} exists")
            continue
        t0 = time.time()
        rows = []
        for lang in args.langs:
            prompts = prompts_by_lang[lang]
            for pid, p in enumerate(prompts):
                if not isinstance(p, str) or not p.strip():
                    rows.append({"prompt_id": pid, "lang": lang, "prompt": p, "response": "",
                                 **cond})
                    continue
                ids = build_chat_ids(tok, p, device)
                if cond["layer"] >= 0 and cond["alpha"] != 0.0:
                    v = torch.from_numpy(v_unit[cond["layer"] + 1].astype(np.float32)).to(device)
                    cm = ResidualSteer(model, cond["layer"], v, cond["alpha"])
                else:
                    cm = _NullCtx()
                with torch.inference_mode(), cm:
                    gen = model.generate(
                        input_ids=ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.eos_token_id,
                    )
                text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
                rows.append({"prompt_id": pid, "lang": lang, "prompt": p, "response": text,
                             **cond})
            print(f"[cond={cond['name']}] lang={lang} done  elapsed={time.time()-t0:.1f}s")
        with open(out_path, "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[save] {out_path}  total_time={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
