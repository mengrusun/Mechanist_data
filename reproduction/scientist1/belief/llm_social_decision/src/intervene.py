"""Claims 3 & 4: causal intervention by adding alpha * direction to residual
stream at a target layer, at every token position. Measure how E[transfer]
depends on the *targeted* variable (specificity) and on alpha sign
(amplify vs invert)."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"
RES = ROOT/"results"


class ResidualAdder:
    def __init__(self, model, layer_idx, vec):
        # vec: torch tensor (H,), same dtype as model residual stream (bf16)
        self.model = model
        self.layer_idx = layer_idx
        self.vec = vec
        self.handle = None

    def _hook(self, module, inp, out):
        # For Llama, the decoder layer output is a tuple (hidden_states, ...) or
        # the module we hook (residual output) may return tensor.
        if isinstance(out, tuple):
            h = out[0]
            h = h + self.vec.to(h.dtype).to(h.device)
            return (h,) + out[1:]
        else:
            return out + self.vec.to(out.dtype).to(out.device)

    def __enter__(self):
        # hook at the output of decoder layer[layer_idx]
        layer = self.model.model.layers[self.layer_idx]
        self.handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        self.handle.remove()


def integer_token_ids(tokenizer):
    ids = []
    for n in range(0, 21):
        tok = tokenizer.encode(str(n), add_special_tokens=False)
        assert len(tok) == 1
        ids.append(tok[0])
    return torch.tensor(ids)


@torch.no_grad()
def score_trials(model, tokenizer, prompts, int_ids, device, batch=32):
    """Return arrays E and argmax for each prompt under the *current* hook state."""
    Es, AMs = [], []
    for start in range(0, len(prompts), batch):
        chunk = prompts[start:start+batch]
        enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        out = model(**enc)
        logits = out.logits[:, -1, :]
        int_logits = logits[:, int_ids]
        probs = torch.softmax(int_logits, dim=-1).float().cpu().numpy()
        for row in probs:
            e = float(np.dot(np.arange(21), row))
            Es.append(e); AMs.append(int(np.argmax(row)))
    return np.array(Es), np.array(AMs)


def main(model_path, dir_dir, trials_path, out_path,
         layer_frac=0.5, alphas=None, use_pure=True, use_raw=True,
         limit=None, batch=32, variables=None):
    device = "cuda"
    if alphas is None:
        alphas = [-6.0, -3.0, -1.5, 0.0, 1.5, 3.0, 6.0]
    print(f"Loading {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    tgt = int(round(n_layers * layer_frac))
    print(f"n_layers={n_layers}, hidden={hidden}, target layer={tgt}")

    int_ids = integer_token_ids(tokenizer).to(device)

    trials = [json.loads(l) for l in open(trials_path)]
    if limit:
        trials = trials[:limit]
    prompts = [t["prompt"] for t in trials]

    dir_dir = Path(dir_dir)
    VARS = variables or ["gender", "age", "instruction", "meeting"]

    # baseline (no hook)
    print("Scoring baseline (no intervention)...")
    E0, AM0 = score_trials(model, tokenizer, prompts, int_ids, device, batch=batch)
    print(f"baseline E mean = {E0.mean():.3f}")

    results = {"target_layer": tgt, "n": len(trials),
               "baseline_E": E0.tolist(),
               "conditions": []}

    # attach labels
    lbl = {
        "gender": np.array([1 if t["gender"]=="male" else 0 for t in trials]),
        "age":    np.array([1 if t["age"]=="old" else 0 for t in trials]),
        "instruction": np.array([1 if t["instruction"]=="B" else 0 for t in trials]),
        "meeting": np.array([1 if t["meeting"]=="meeting" else 0 for t in trials]),
    }

    def summary(E):
        # per-variable difference in mean E[transfer] between the two values
        s = {}
        for v in ["gender","age","instruction","meeting"]:
            y = lbl[v]
            s[v] = float(E[y==1].mean() - E[y==0].mean())
        s["mean"] = float(E.mean())
        return s

    S0 = summary(E0)
    results["baseline_summary"] = S0
    print("baseline effects:", S0)

    kinds = []
    if use_raw:  kinds.append(("raw", "dir"))
    if use_pure: kinds.append(("pure","pure"))

    for kind, prefix in kinds:
        for v in VARS:
            dvec = np.load(dir_dir/f"{prefix}_{v}.npy")[tgt]  # (H,)
            dvec_unit = dvec / (np.linalg.norm(dvec)+1e-9)
            for a in alphas:
                if a == 0.0:
                    continue
                vec = torch.from_numpy((a * dvec_unit).astype(np.float32)).to(device)
                with ResidualAdder(model, tgt, vec):
                    E, _ = score_trials(model, tokenizer, prompts, int_ids, device, batch=batch)
                S = summary(E)
                results["conditions"].append({
                    "kind": kind, "variable": v, "alpha": a,
                    "layer": tgt,
                    "summary": S,
                    "delta_summary_vs_base": {k: S[k]-S0[k] for k in S},
                })
                print(f"{kind:4s} {v:11s} a={a:+.2f} -> gender={S['gender']:+.3f} age={S['age']:+.3f} instr={S['instruction']:+.3f} meet={S['meeting']:+.3f} mean={S['mean']:.3f}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhenqian/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--dir_dir", default=str(RES/"directions_llama"))
    ap.add_argument("--trials", default=str(DATA/"trials.jsonl"))
    ap.add_argument("--out",    default=str(RES/"intervention_llama.json"))
    ap.add_argument("--layer_frac", type=float, default=0.5)
    ap.add_argument("--alphas", default="-6,-3,-1.5,0,1.5,3,6")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pure_only", action="store_true")
    ap.add_argument("--raw_only",  action="store_true")
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",")]
    main(args.model, args.dir_dir, args.trials, args.out,
         layer_frac=args.layer_frac, alphas=alphas,
         use_pure=not args.raw_only, use_raw=not args.pure_only,
         limit=args.limit, batch=args.batch)
