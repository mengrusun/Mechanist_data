"""Steering-intervention experiment to test dissociation of harmfulness/refusal directions.

For each intervention condition we:
  1. Add α * v (direction, unit-norm) to the residual stream after layer L.
  2. Extract post-intervention hidden states at (L+1 ... last_layer) to measure
     projection changes.
  3. Generate the model's response.

Conditions:
  * baseline (no steer) — benign prompts
  * baseline (no steer) — harmful prompts
  * +v_harm on benign  (should flip 'internal harmfulness' up; refusal ?)
  * -v_harm on harmful (should flip 'internal harmfulness' down; refusal ?)
  * +v_refuse on benign (should induce refusal; internal harm ?)
  * -v_refuse on harmful (should suppress refusal; internal harm ?)
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    LLAMA3_PATH, load_model_and_tokenizer, format_llama3_chat, is_refusal,
    load_advbench, load_alpaca_benign, T_INST, T_POST,
)

RES_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/results"


class Steerer:
    """Adds α * v to the residual stream at a given layer.

    Uses a forward hook on the decoder block: adds `alpha * v_unit` to the output.
    Handles both forward for hidden-state extraction AND generation.
    """
    def __init__(self, model, layer, v_unit, alpha):
        self.model = model
        self.layer = layer
        self.v = torch.as_tensor(v_unit, dtype=next(model.parameters()).dtype,
                                  device=next(model.parameters()).device)
        self.alpha = alpha
        self.hook_handle = None

    def _hook(self, module, inputs, output):
        # output is either (hidden_states, ...) tuple or just tensor
        if isinstance(output, tuple):
            hs = output[0]
            v = self.v.to(hs.device, hs.dtype)
            hs = hs + self.alpha * v
            return (hs,) + output[1:]
        else:
            v = self.v.to(output.device, output.dtype)
            return output + self.alpha * v

    def __enter__(self):
        # Locate transformer block: model.model.layers[layer]
        block = self.model.model.layers[self.layer]
        self.hook_handle = block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *args):
        if self.hook_handle is not None:
            self.hook_handle.remove()


@torch.no_grad()
def forward_and_extract(model, tok, prompts, layers_to_read, positions=(T_INST, T_POST),
                        batch_size=8, device="cuda"):
    """Return dict[(layer, pos)] -> np.ndarray (N, D)."""
    D = model.config.hidden_size
    N = len(prompts)
    buf = {(l, p): np.empty((N, D), dtype=np.float32) for l in layers_to_read for p in positions}
    for i in range(0, N, batch_size):
        batch = prompts[i:i+batch_size]
        formatted = [format_llama3_chat(tok, p) for p in batch]
        enc = tok(formatted, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        for l in layers_to_read:
            h = out.hidden_states[l]
            for p in positions:
                buf[(l, p)][i:i+len(batch)] = h[:, p].float().cpu().numpy()
    return buf


@torch.no_grad()
def generate_all(model, tok, prompts, max_new_tokens=96, batch_size=8, device="cuda"):
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        formatted = [format_llama3_chat(tok, p) for p in batch]
        enc = tok(formatted, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.pad_token_id,
        )
        for j in range(len(batch)):
            in_len = enc.input_ids.shape[1]
            outs.append(tok.decode(gen[j, in_len:], skip_special_tokens=True))
    return outs


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of samples of each type")
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--layer", type=int, default=11, help="layer at which we intervene")
    ap.add_argument("--read_layer", type=int, default=20, help="layer at which we read the residual")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    args = ap.parse_args()

    # Load directions (per layer, D)
    d = np.load(os.path.join(RES_DIR, "directions_llama3.npz"))
    # unit-norm directions at the intervention layer
    v_inst = unit(d["v_inst"][args.layer])   # harmfulness direction (measured at t_inst)
    v_post = unit(d["v_post"][args.layer])   # refusal direction (measured at t_post)

    # Also load directions at read_layer for probing "internal state"
    v_inst_r = unit(d["v_inst"][args.read_layer])
    v_post_r = unit(d["v_post"][args.read_layer])

    print("Loading Llama-3-8B-Instruct ...")
    model, tok = load_model_and_tokenizer(LLAMA3_PATH)
    device = next(model.parameters()).device

    benign  = load_alpaca_benign(n=args.n, seed=5)
    harmful = load_advbench(n=args.n, seed=5)

    conditions = [
        ("benign_baseline",         benign,  None,     0.0),
        ("harmful_baseline",        harmful, None,     0.0),
        ("benign_plus_vharm",       benign,  v_inst,  +args.alpha),
        ("harmful_minus_vharm",     harmful, v_inst,  -args.alpha),
        ("benign_plus_vrefuse",     benign,  v_post,  +args.alpha),
        ("harmful_minus_vrefuse",   harmful, v_post,  -args.alpha),
    ]

    results = {"alpha": args.alpha, "layer": args.layer, "read_layer": args.read_layer, "n": args.n, "conditions": {}}

    layers_to_read = [args.read_layer]
    positions = (T_INST, T_POST)

    for name, prompts, v, alpha in conditions:
        print(f"\n=== {name} (alpha={alpha}) ===")
        t0 = time.time()
        if v is None:
            buf = forward_and_extract(model, tok, prompts, layers_to_read, positions,
                                       batch_size=args.batch_size, device=device)
            gens = generate_all(model, tok, prompts, max_new_tokens=args.max_new_tokens,
                                 batch_size=args.batch_size, device=device)
        else:
            with Steerer(model, args.layer, v, alpha):
                buf = forward_and_extract(model, tok, prompts, layers_to_read, positions,
                                           batch_size=args.batch_size, device=device)
                gens = generate_all(model, tok, prompts, max_new_tokens=args.max_new_tokens,
                                     batch_size=args.batch_size, device=device)
        # Measurements: mean projection onto (v_inst_r, v_post_r) at (read_layer, T_INST) and (read_layer, T_POST)
        proj_harm_at_inst   = buf[(args.read_layer, T_INST)] @ v_inst_r
        proj_refuse_at_post = buf[(args.read_layer, T_POST)] @ v_post_r
        proj_harm_at_post   = buf[(args.read_layer, T_POST)] @ v_inst_r
        proj_refuse_at_inst = buf[(args.read_layer, T_INST)] @ v_post_r
        refuse_rate = float(np.mean([is_refusal(o) for o in gens]))
        results["conditions"][name] = {
            "proj_harm_at_inst_mean":   float(proj_harm_at_inst.mean()),
            "proj_harm_at_inst_std":    float(proj_harm_at_inst.std()),
            "proj_refuse_at_post_mean": float(proj_refuse_at_post.mean()),
            "proj_refuse_at_post_std":  float(proj_refuse_at_post.std()),
            "proj_harm_at_post_mean":   float(proj_harm_at_post.mean()),
            "proj_refuse_at_inst_mean": float(proj_refuse_at_inst.mean()),
            "refusal_rate_regex":       refuse_rate,
            "sample_outputs":           gens[:3],
        }
        print(f"  proj_harm@inst mean = {proj_harm_at_inst.mean():+.3f}")
        print(f"  proj_refuse@post mean = {proj_refuse_at_post.mean():+.3f}")
        print(f"  refusal rate = {refuse_rate:.3f}   ({time.time()-t0:.1f}s)")

    out_fp = os.path.join(RES_DIR, "steering.json")
    with open(out_fp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_fp}")


if __name__ == "__main__":
    main()
