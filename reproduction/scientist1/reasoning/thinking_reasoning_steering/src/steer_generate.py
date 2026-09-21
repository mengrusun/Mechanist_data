"""
Steer generation of DeepSeek-R1-Distill-Llama-8B by adding a steering vector
to the residual stream at a chosen layer, and sweep the scalar coefficient.

Usage:
  python steer_generate.py \
     --directions data/directions.pt \
     --probe results/probe_accuracy.json \
     --problems data/eval_problems.json \
     --alphas -3 -2 -1 0 1 2 3 \
     --out results/steered_generations.json
"""
import os, json, argparse, gc, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B"


class SteerHook:
    """Adds alpha * unit_direction to the residual stream at target layer for every position."""
    def __init__(self, unit_dir, alpha):
        self.unit_dir = unit_dir
        self.alpha = alpha
        self.h = None
    def __enter__(self):
        return self
    def __exit__(self, *a):
        if self.h is not None: self.h.remove()
    def attach(self, module):
        d = self.unit_dir.to(next(module.parameters()).device)
        alpha = self.alpha
        def hook(mod, inp, out):
            # decoder layer output is either a tensor or a tuple whose first element is the hidden state
            if isinstance(out, tuple):
                h = out[0]
                h = h + alpha * d.to(h.dtype)
                return (h,) + out[1:]
            else:
                return out + alpha * d.to(out.dtype)
        self.h = module.register_forward_hook(hook)


@torch.no_grad()
def generate_steered(model, tokenizer, problem, steer_layer_idx, unit_dir, alpha,
                     max_new_tokens=350, temperature=0.7, top_p=0.95, seed=0):
    torch.manual_seed(seed)
    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    # target layer = model.model.layers[i]   (residual stream after that block)
    target = model.model.layers[steer_layer_idx]

    hook = SteerHook(unit_dir, alpha)
    hook.attach(target)
    try:
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        hook.__exit__(None, None, None)

    gen = out_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", default="data/directions.pt")
    ap.add_argument("--probe",      default="results/probe_accuracy.json")
    ap.add_argument("--problems",   default="data/eval_problems.json")
    ap.add_argument("--alphas", type=float, nargs="+", default=[-8.0,-4.0,-2.0,0.0,2.0,4.0,8.0])
    ap.add_argument("--out", default="results/steered_generations.json")
    ap.add_argument("--max_new_tokens", type=int, default=350)
    ap.add_argument("--behaviours", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    dtype  = torch.bfloat16

    print("[load] tokenizer + model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype, device_map={"": device})
    model.eval()

    directions = torch.load(args.directions, weights_only=False)
    with open(args.probe) as f:
        probe = json.load(f)
    with open(args.problems) as f:
        problems = json.load(f)
    # accept either list[str] or list[{problem: str, ...}]
    problems = [p if isinstance(p, str) else p["problem"] for p in problems]

    behaviours = args.behaviours or list(directions.keys())

    # Pick the best MID layer per behaviour by val_acc. We restrict to layers
    # in [8, 20] to avoid (a) early layers where the direction is noisy, and
    # (b) late layers where residual-stream norms are huge and additive steering
    # tends to derail generation. Among the restricted range pick the layer
    # with highest val_acc; ties broken by earliest layer.
    def pick_best_layer(b):
        candidates = [(int(L), probe[b][str(L)]["val_acc"]) for L in range(8, 21)]
        best_acc = max(a for _, a in candidates)
        for L, a in candidates:
            if a == best_acc:
                return L

    steer_layers = {b: pick_best_layer(b) for b in behaviours}
    print(f"[steer] target layers: {steer_layers}")

    all_generations = []
    for pi, problem in enumerate(problems):
        for b in behaviours:
            L = steer_layers[b]
            v = directions[b][L]
            unit = v / (v.norm() + 1e-8)
            for alpha in args.alphas:
                text = generate_steered(model, tokenizer, problem,
                                        steer_layer_idx=L - 1,  # residual after that block -> hook on layer index
                                        unit_dir=unit,
                                        alpha=alpha,
                                        max_new_tokens=args.max_new_tokens,
                                        seed=args.seed)
                all_generations.append({
                    "problem": problem,
                    "behaviour": b,
                    "layer": L,
                    "alpha": alpha,
                    "text": text,
                })
                print(f"[{pi+1}/{len(problems)}] {b} L={L} α={alpha:+.2f} → {len(text)} chars", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"steer_layers": steer_layers, "alphas": args.alphas, "generations": all_generations}, f, indent=2)
    print(f"[save] wrote {args.out}")


if __name__ == "__main__":
    main()
