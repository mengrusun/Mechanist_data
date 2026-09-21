#!/usr/bin/env python3
"""M2 / M2b / M3: per-condition prefix evaluation on Qwen3-14B.

Two-phase design:
  Phase A: run vLLM generation for all items at temperature 0.0 (fast, batched)
  Phase B: run one HF forward pass per item to capture last-prefix-token
           residual activations (fast because we only need one forward,
           and we batch across items with left-padding on shared prefix)

For MCQ (task in {socialiqa, medqa}): use vLLM's log-prob API on the letter tokens.
Runs for one (task, condition_id) at a time.

Output:
  runs/<milestone>/<task>_<condition_id>.json  — per-item correctness log
  runs/<milestone>/act_<task>_<condition_id>.pt — dict {layer:int -> tensor[n_items, d_model]}
  runs/<milestone>/cost.json — GPU-hours + gpu_ids (per-run key)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch


# ----- Data loaders --------------------------------------------------------

def load_gsm8k(n_items: int) -> List[Dict]:
    from datasets import load_from_disk
    ds = load_from_disk("/data/zhenqian/data/gsm8k")["test"]
    out = []
    for i, r in enumerate(ds):
        if i >= n_items:
            break
        m = re.search(r"####\s*(-?[\d,]+)", r["answer"])
        gold = m.group(1).replace(",", "") if m else None
        out.append({"item_id": i, "question": r["question"], "gold": gold})
    return out


def load_socialiqa(n_items: int) -> List[Dict]:
    root = "/data/zhenqian/data/social_i_qa_full/socialiqa-train-dev"
    lines = open(f"{root}/dev.jsonl").readlines()
    labels = open(f"{root}/dev-labels.lst").read().strip().splitlines()
    out = []
    for i, (ln, lab) in enumerate(zip(lines, labels)):
        if i >= n_items:
            break
        r = json.loads(ln)
        out.append({
            "item_id": i,
            "context": r["context"],
            "question": r["question"],
            "options": {"A": r["answerA"], "B": r["answerB"], "C": r["answerC"]},
            "gold": {"1": "A", "2": "B", "3": "C"}[lab.strip()],
        })
    return out


def load_medqa(n_items: int) -> List[Dict]:
    path = "/data/zhenqian/data/med_qa/data_clean/questions/US/test.jsonl"
    out = []
    for i, ln in enumerate(open(path)):
        if i >= n_items:
            break
        r = json.loads(ln)
        out.append({
            "item_id": i,
            "question": r["question"],
            "options": r["options"],
            "gold": r["answer_idx"],
        })
    return out


# ----- Prompt builders -----------------------------------------------------

GSM8K_INSTRUCTION = (
    "Solve the math problem step by step, then write the final answer as an integer "
    'after a "####" marker on a new line. Example: "#### 42".\n\n'
)


def build_gsm8k_prompt(prefix: str, item: Dict) -> str:
    return (
        prefix.strip() + "\n\n"
        + GSM8K_INSTRUCTION
        + "Problem: " + item["question"].strip() + "\n"
        + "Solution:"
    )


def build_mcq_prompt(prefix: str, item: Dict, task: str) -> str:
    """Prompt ending with 'Answer:' — we score next-token log-prob of the letter."""
    if task == "socialiqa":
        opts_text = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
        body = (
            f"Context: {item['context'].strip()}\n"
            f"Question: {item['question'].strip()}\n"
            f"Options:\n{opts_text}\n"
        )
    else:  # medqa
        opts_text = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
        body = (
            f"Question: {item['question'].strip()}\n"
            f"Options:\n{opts_text}\n"
        )
    return (
        prefix.strip() + "\n\n"
        + "Read the question and options; then answer with just the single option letter.\n\n"
        + body
        + "Answer:"
    )


# ----- Answer parsing ------------------------------------------------------

_GSM_RE = re.compile(r"####\s*(-?[\d,]+)")


def parse_gsm8k(text: str) -> Optional[str]:
    m = _GSM_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    ints = re.findall(r"-?\d+", text.replace(",", ""))
    return ints[-1] if ints else None


# ----- vLLM generation phase -----------------------------------------------

def vllm_generate(model_dir: str, prompts: List[str], max_new_tokens: int,
                  gpu_ids: str, mcq_mode: bool = False,
                  option_letters_per_item: Optional[List[List[str]]] = None,
                  ) -> tuple:
    """Run vLLM inference. Returns (gen_texts_or_None, mcq_preds_or_None, mcq_lps_or_None)."""
    from vllm import LLM, SamplingParams

    # Configure GPU count
    n_gpus = len(gpu_ids.split(",")) if gpu_ids and gpu_ids != "auto" else 1
    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        tensor_parallel_size=n_gpus,
        gpu_memory_utilization=0.60,
        enable_prefix_caching=False,   # per plan: prevent inter-condition contamination
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True,   # skip CUDA graph capture (saves ~10s startup, small runtime penalty)
    )
    tok = llm.get_tokenizer()

    if not mcq_mode:
        sp = SamplingParams(
            temperature=0.0, top_p=1.0,
            max_tokens=max_new_tokens,
            # Stop as soon as the model emits a second "Problem:" line (starts a new question).
            stop=["\n\nProblem:", "\nProblem:", "\n\nQuestion:"],
        )
        outputs = llm.generate(prompts, sp)
        gen_texts = [o.outputs[0].text for o in outputs]
        gen_tokens = [len(o.outputs[0].token_ids) for o in outputs]
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        return gen_texts, gen_tokens, None
    else:
        # MCQ: 1-token generation with log-probs
        # Prepare per-prompt letter tokens
        assert option_letters_per_item is not None
        # Get single-token ids for each letter (with leading space + without)
        letter_token_ids: Dict[str, int] = {}
        for L in "ABCDE":
            for cand in [f" {L}", L]:
                ids = tok.encode(cand, add_special_tokens=False)
                if len(ids) == 1:
                    if L not in letter_token_ids:
                        letter_token_ids[L] = ids[0]
                    break
        sp = SamplingParams(
            temperature=0.0, max_tokens=1, logprobs=20,
        )
        outputs = llm.generate(prompts, sp)
        preds: List[str] = []
        letter_lps: List[Dict[str, float]] = []
        for out, letters in zip(outputs, option_letters_per_item):
            lp_dict = {}
            # vllm 0.10 API: out.outputs[0].logprobs is a list-of-dicts (one per gen token)
            lp = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            for L in letters:
                tid = letter_token_ids.get(L, None)
                if tid is None:
                    lp_dict[L] = -1e9
                elif tid in lp:
                    lp_dict[L] = float(lp[tid].logprob)
                else:
                    lp_dict[L] = -1e9
            preds.append(max(lp_dict, key=lp_dict.get))
            letter_lps.append(lp_dict)
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        return None, preds, letter_lps


# ----- HF activation-capture phase ----------------------------------------

def hf_capture_activations(model_dir: str, prompts_for_cap: List[str],
                            prefix_ends: List[int],
                            layers: List[int], batch_size: int = 8) -> Dict[int, torch.Tensor]:
    """Run one HF forward pass per prompt to capture residual activations at
    prefix-end token for each requested layer.

    prompts_for_cap: list of prompts (typically the same as for generation,
                     but truncated at the prefix end to save memory).
    prefix_ends:     list of int token counts — the token index of the last
                     prefix token in each prompt.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[hf] Loading model for activation capture ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    # Per-layer accumulator
    per_layer: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}

    # We batch prompts; prefix_ends may differ. Use left-padding so
    # positions align.
    for start in range(0, len(prompts_for_cap), batch_size):
        batch_prompts = prompts_for_cap[start:start + batch_size]
        batch_ends = prefix_ends[start:start + batch_size]
        # Encode individually to get exact positions
        for prompt_str, prefix_end_tok_idx in zip(batch_prompts, batch_ends):
            input_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=True
                                  ).input_ids.to(model.device)
            # The prefix_end_tok_idx was computed against tokenizer w/ same conventions
            pos = min(prefix_end_tok_idx, input_ids.shape[1] - 1)
            with torch.no_grad():
                out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            for L in layers:
                per_layer[L].append(out.hidden_states[L][0, pos, :].detach().float().cpu())
            del out

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {L: torch.stack(v) for L, v in per_layer.items() if v}


# ----- Main ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gsm8k", "socialiqa", "medqa"], required=True)
    ap.add_argument("--model", default="/data/zhenqian/models/Qwen3-14B")
    ap.add_argument("--prefixes_json", default="data/prefixes/prefixes.json")
    ap.add_argument("--noise_json", default="data/prefixes/format_noise_variants.json")
    ap.add_argument("--condition_id", required=True)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--eval_mode", choices=["cot", "mcq_ll"], default="cot")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cache_residual_layers", default="0,4,8,12,16,20,24,28,32,36")
    ap.add_argument("--out", required=True)
    ap.add_argument("--activations_out", default=None)
    ap.add_argument("--skip_activations", action="store_true")
    ap.add_argument("--activations_n", type=int, default=None,
                    help="If set, cap number of items for activation capture "
                         "(e.g., 200 for M5's paired subset)")
    ap.add_argument("--gpu_ids", default=os.environ.get("CUDA_VISIBLE_DEVICES", "auto"))
    args = ap.parse_args()

    print(f"[cfg] task={args.task} condition={args.condition_id} n_items={args.n_items} "
          f"eval_mode={args.eval_mode} gpu={args.gpu_ids}")
    layers = [int(x) for x in args.cache_residual_layers.split(",") if x.strip()]

    prefixes = json.load(open(args.prefixes_json))
    noise = json.load(open(args.noise_json))
    all_prefixes = {r["condition_id"]: r["text"] for r in prefixes}
    all_prefixes.update({r["condition_id"]: r["text"] for r in noise})
    if args.condition_id not in all_prefixes:
        raise ValueError(f"Unknown condition_id={args.condition_id}")
    prefix_text = all_prefixes[args.condition_id]

    # Load data
    if args.task == "gsm8k":
        items = load_gsm8k(args.n_items)
    elif args.task == "socialiqa":
        items = load_socialiqa(args.n_items)
    else:
        items = load_medqa(args.n_items)

    # Build prompts
    if args.eval_mode == "cot":
        prompts = [build_gsm8k_prompt(prefix_text, it) for it in items]
        option_letters_per_item = None
    else:
        prompts = [build_mcq_prompt(prefix_text, it, args.task) for it in items]
        option_letters_per_item = [list(it["options"].keys()) for it in items]

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ----- Phase A: vLLM inference -----
    if args.eval_mode == "cot":
        gen_texts, gen_tokens, _ = vllm_generate(
            args.model, prompts, args.max_new_tokens, args.gpu_ids, mcq_mode=False,
        )
        # Parse & score
        results = []
        for it, text, ntok in zip(items, gen_texts, gen_tokens):
            pred = parse_gsm8k(text)
            gold = it["gold"]
            try:
                correct = int(pred is not None and gold is not None and float(pred) == float(gold))
            except Exception:
                correct = int(pred == gold) if (pred is not None and gold is not None) else 0
            parse_ok = int(pred is not None)
            results.append({
                "item_id": it["item_id"], "gold": gold, "pred": pred,
                "correct": correct, "parse_ok": parse_ok,
                "n_gen_tokens": ntok, "gen_text": text,
            })
    else:
        _, preds, letter_lps = vllm_generate(
            args.model, prompts, 1, args.gpu_ids, mcq_mode=True,
            option_letters_per_item=option_letters_per_item,
        )
        results = []
        for it, pred, lp in zip(items, preds, letter_lps):
            gold = it["gold"]
            correct = int(pred == gold)
            results.append({
                "item_id": it["item_id"], "gold": gold, "pred": pred,
                "correct": correct, "letter_lp": lp,
            })

    n_correct = sum(r["correct"] for r in results)
    acc = n_correct / len(results)
    print(f"[phase A] acc={acc:.4f} ({n_correct}/{len(results)}) t={time.time()-t0:.1f}s")

    # ----- Phase B: HF activation capture -----
    acts_tensor = {}
    if not args.skip_activations and args.activations_out:
        # Only capture activations for the first args.activations_n items
        # (defaults to all if not set)
        cap_n = args.activations_n if args.activations_n else len(items)
        cap_prompts = prompts[:cap_n]

        # Compute prefix-end token index for each prompt
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        prefix_ends = []
        for p in cap_prompts:
            prefix_end_char = p.find(prefix_text) + len(prefix_text)
            prefix_slice = p[:prefix_end_char]
            n_tok = len(tokenizer(prefix_slice, add_special_tokens=True).input_ids)
            prefix_ends.append(n_tok - 1)

        acts_tensor = hf_capture_activations(
            args.model, cap_prompts, prefix_ends, layers, batch_size=1,
        )
        cap_ids = [it["item_id"] for it in items[:cap_n]]

    elapsed = time.time() - t0
    print(f"[done] {args.condition_id} acc={acc:.4f} time={elapsed:.1f}s")

    summary = {
        "condition_id": args.condition_id,
        "task": args.task,
        "n_items": len(items),
        "accuracy": acc,
        "n_correct": n_correct,
        "prefix_text": prefix_text,
        "eval_mode": args.eval_mode,
        "elapsed_seconds": elapsed,
        "per_item": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[write] {args.out}")

    if acts_tensor and args.activations_out:
        Path(os.path.dirname(args.activations_out)).mkdir(parents=True, exist_ok=True)
        torch.save({"condition_id": args.condition_id, "task": args.task,
                    "layers": layers, "acts": acts_tensor,
                    "item_ids": cap_ids}, args.activations_out)
        print(f"[write] {args.activations_out}  layers={list(acts_tensor.keys())}  "
              f"shape={acts_tensor[layers[0]].shape}")

    # cost.json
    run_dir = os.path.dirname(args.out)
    cost_path = os.path.join(run_dir, "cost.json")
    gpu_ids_list = ([int(x) for x in args.gpu_ids.split(",")]
                    if args.gpu_ids and args.gpu_ids != "auto" else [])
    key = f"{args.task}_{args.condition_id}"
    cost = {
        "run_id": key,
        "gpu_ids": gpu_ids_list,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0 * max(1, len(gpu_ids_list) if gpu_ids_list else 1),
    }
    old = {}
    if os.path.exists(cost_path):
        try:
            old = json.load(open(cost_path))
        except Exception:
            old = {}
    if not isinstance(old, dict):
        old = {}
    old[key] = cost
    with open(cost_path, "w") as f:
        json.dump(old, f, indent=1)


if __name__ == "__main__":
    main()
