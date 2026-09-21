"""
Prompt-engineering baseline: instead of a steering vector, prepend/append a
natural-language directive that asks the model to exhibit or avoid a behaviour.
Used for Claim 4: steering is finer-grained than prompting.
"""
import os, json, argparse, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B"

DIRECTIVES = {
    "uncertainty": {
        "amplify":  "As you reason, frequently express uncertainty and hedge with phrases like 'I'm not sure', 'maybe', 'I could be wrong', 'let me double check'.",
        "suppress": "As you reason, be maximally confident and direct. Never hedge or express uncertainty; state each step as a definite fact.",
    },
    "example_generation": {
        "amplify":  "As you reason, constantly test with concrete small examples (e.g., 'let me try n=3', 'plug in x=2 and check').",
        "suppress": "As you reason, work purely with abstract symbols and formulas. Never test with concrete example values.",
    },
    "backtracking": {
        "amplify":  "As you reason, try one approach first, then backtrack and switch approach at least once with phrases like 'wait, that's wrong' or 'let me reconsider'.",
        "suppress": "As you reason, proceed in a single clean linear chain of thought without ever backtracking, second-guessing, or switching approach.",
    },
}


@torch.no_grad()
def generate(model, tokenizer, problem, directive=None, max_new_tokens=350, temperature=0.7, top_p=0.95, seed=0):
    torch.manual_seed(seed)
    if directive:
        messages = [{"role": "system", "content": directive},
                    {"role": "user", "content": problem}]
    else:
        messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**inputs,
                         max_new_tokens=max_new_tokens,
                         do_sample=(temperature > 0),
                         temperature=temperature if temperature > 0 else 1.0,
                         top_p=top_p,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="data/eval_problems.json")
    ap.add_argument("--out", default="results/prompt_baseline.json")
    ap.add_argument("--behaviours", nargs="+", default=list(DIRECTIVES.keys()))
    ap.add_argument("--max_new_tokens", type=int, default=350)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    dtype  = torch.bfloat16
    print("[load] tokenizer + model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype, device_map={"": device})
    model.eval()

    with open(args.problems) as f:
        probs = json.load(f)

    gens = []
    for pi, p in enumerate(probs):
        prob = p["problem"] if isinstance(p, dict) else p
        # baseline: no directive
        base = generate(model, tokenizer, prob, directive=None,
                        max_new_tokens=args.max_new_tokens, seed=args.seed)
        gens.append({"problem": prob, "behaviour": "none", "condition": "baseline",
                     "text": base})
        for b in args.behaviours:
            for cond in ("amplify", "suppress"):
                text = generate(model, tokenizer, prob, directive=DIRECTIVES[b][cond],
                                max_new_tokens=args.max_new_tokens, seed=args.seed)
                gens.append({"problem": prob, "behaviour": b, "condition": cond,
                             "text": text})
        print(f"[{pi+1}/{len(probs)}] done", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"generations": gens}, f, indent=2)
    print(f"[save] wrote {args.out}")


if __name__ == "__main__":
    main()
