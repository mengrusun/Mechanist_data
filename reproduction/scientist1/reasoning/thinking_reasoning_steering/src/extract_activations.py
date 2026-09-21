"""
Extract residual-stream activations from DeepSeek-R1-Distill-Llama-8B on
contrastive pairs, and save one tensor per (behaviour, split, condition, layer).

Design:
  - Prompt format: <|begin_of_text|><|User|>{problem}<|Assistant|><think>{prefix} {continuation}
  - We tokenize this once and take the residual stream AT THE LAST TOKEN of the
    continuation for every hidden layer.
  - Contrastive: same problem + prefix, only continuation differs -> "present" vs "absent".
  - We hold out 20% of the pairs (per behaviour) as validation for the linear-probe test.
"""
import os, json, argparse, random, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B"


def format_prompt(tokenizer, problem, prefix_and_continuation):
    """Compose a chat-style prompt with the CoT continuation appended inside <think>.
    Returns input_ids (1, T) and the position of the last token (=last token of continuation)."""
    # DeepSeek-R1 uses <｜User｜> / <｜Assistant｜> and <think>...</think>.
    # We mimic that style using the tokenizer's chat template if available,
    # else fall back to a manual template.
    system_user = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # `system_user` ends with something like "<|Assistant|>" — append <think> and continuation.
    text = system_user + "<think>\n" + prefix_and_continuation
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    return input_ids


@torch.no_grad()
def extract_layer_acts(model, tokenizer, texts_pairs, device, dtype):
    """
    texts_pairs: list of dicts with keys 'problem', 'prefix', 'present', 'absent'.
    Returns:
        acts_present: (N, L+1, H)  last-token hidden state per layer
        acts_absent:  (N, L+1, H)
        # H = hidden_size, L = num layers. hidden_states[0] is embeddings, then L transformer outputs.
    """
    n_layers_incl_emb = model.config.num_hidden_layers + 1
    H = model.config.hidden_size
    acts_present = torch.zeros(len(texts_pairs), n_layers_incl_emb, H, dtype=torch.float32)
    acts_absent  = torch.zeros(len(texts_pairs), n_layers_incl_emb, H, dtype=torch.float32)

    for i, p in enumerate(texts_pairs):
        for cond, buf in (("present", acts_present), ("absent", acts_absent)):
            cont = p["prefix"] + " " + p[cond]
            input_ids = format_prompt(tokenizer, p["problem"], cont).to(device)
            out = model(input_ids, output_hidden_states=True, use_cache=False)
            # out.hidden_states: tuple of (n_layers+1) tensors of shape (1, T, H)
            for L, h in enumerate(out.hidden_states):
                buf[i, L] = h[0, -1].float().cpu()
            del out
        if (i+1) % 5 == 0:
            print(f"  [extract] {i+1}/{len(texts_pairs)}", flush=True)
    return acts_present, acts_absent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/contrastive_pairs.json")
    ap.add_argument("--out", default="data/activations.pt")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    dtype  = torch.bfloat16

    print("[load] tokenizer + model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype, device_map={"": device})
    model.eval()
    print(f"[load] hidden_size={model.config.hidden_size} num_layers={model.config.num_hidden_layers}")

    with open(args.pairs) as f:
        pairs = json.load(f)

    out = {"config": {"model": MODEL_PATH, "hidden_size": model.config.hidden_size,
                       "num_layers": model.config.num_hidden_layers,
                       "val_frac": args.val_frac, "seed": args.seed}}

    for behaviour, ps in pairs.items():
        random.Random(args.seed).shuffle(ps)
        n_val = max(1, int(len(ps) * args.val_frac))
        val = ps[:n_val]
        trn = ps[n_val:]
        print(f"[{behaviour}] train={len(trn)} val={len(val)}")

        print(f"[{behaviour}] extracting train activations...")
        tp, ta = extract_layer_acts(model, tokenizer, trn, device, dtype)
        print(f"[{behaviour}] extracting val activations...")
        vp, va = extract_layer_acts(model, tokenizer, val, device, dtype)

        out[behaviour] = {
            "train_present": tp, "train_absent": ta,
            "val_present":   vp, "val_absent":   va,
            "train_pairs":   trn, "val_pairs": val,
        }
        gc.collect(); torch.cuda.empty_cache()

    torch.save(out, args.out)
    print(f"[save] wrote {args.out}")

if __name__ == "__main__":
    main()
