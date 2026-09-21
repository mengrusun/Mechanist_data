"""M4 — Evaluate a (possibly LoRA-adapted) model on MGSM 11 languages.

Same evaluation protocol as M2 but *without* any residual-stream steering hook — this is the "training baseline"
side. Supports:
    --lora_adapter <path>   : load a PEFT LoRA adapter on top of the base model.
    --no_lora               : just evaluate the base model.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlr.data import MGSM_LANGS, GLOTLID_MAP, load_mgsm_all, default_few_shot_en, format_mgsm_prompt, normalize_langs
from mlr.mgsm_eval import extract_answer, numeric_equal
from mlr.m2_projection_eval import _hf_generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--lora_adapter", default=None)
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--languages", default=",".join(MGSM_LANGS))
    ap.add_argument("--n_shot", type=int, default=3)
    ap.add_argument("--n_problems_per_lang", type=int, default=250)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--glotlid_path", default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--greedy", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True,
    )
    model.eval()

    if args.lora_adapter and not args.no_lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_adapter)
        model.eval()
        print(f"[m4] loaded LoRA adapter from {args.lora_adapter}", flush=True)

    raw_langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    langs = normalize_langs(raw_langs)
    mgsm = load_mgsm_all(langs=langs, split="test", data_dir=args.data_dir)

    glot = None
    try:
        from mlr.glotlid_wrapper import GlotLID
        glot = GlotLID(model_path=args.glotlid_path)
    except Exception as e:
        print(f"[m4] GlotLID unavailable: {e}", flush=True)

    fewshot = default_few_shot_en()[: args.n_shot]
    records = []
    per_lang = {}
    for lang in langs:
        df = mgsm[lang]
        n = min(args.n_problems_per_lang, len(df))
        prompts = []
        golds = []
        for i in range(n):
            q = str(df.iloc[i]["question"])
            gold = df.iloc[i].get("answer_number", df.iloc[i].get("answer", None))
            prompts.append(format_mgsm_prompt(q, fewshot))
            golds.append(gold)
        gens = _hf_generate(model, tokenizer, prompts, args.max_new_tokens, args.batch_size, args.greedy)
        n_correct = 0
        n_fluent = 0
        for j in range(n):
            gold_n = None
            try:
                if golds[j] is not None and str(golds[j]).strip() != "":
                    gold_n = float(str(golds[j]).replace(",", ""))
            except Exception:
                gold_n = None
            pred = extract_answer(gens[j])
            is_correct = numeric_equal(pred, gold_n) if gold_n is not None else False
            fid_pred = fid_score = fid_correct = None
            if glot is not None:
                fid_pred, fid_score = glot.predict(gens[j])
                fid_correct = int(fid_pred == GLOTLID_MAP.get(lang, ""))
                if fid_correct:
                    n_fluent += 1
            records.append({"lang": lang, "problem_id": int(df.iloc[j].get("id", j)),
                            "question": str(df.iloc[j]["question"]), "gold": gold_n,
                            "generation": gens[j], "extracted": pred, "is_correct": bool(is_correct),
                            "glotlid_pred": fid_pred, "glotlid_score": fid_score,
                            "fidelity_correct": fid_correct})
            if is_correct:
                n_correct += 1
        acc = n_correct / max(n, 1)
        fid = (n_fluent / max(n, 1)) if glot is not None else None
        per_lang[lang] = {"n": n, "accuracy": acc, "fidelity": fid}
        print(f"[m4] {lang}: acc={acc:.3f}  fid={fid if fid is None else round(fid,3)}", flush=True)

    macro = float(np.mean([v["accuracy"] for v in per_lang.values()]))
    macro_fid = None
    fids = [v["fidelity"] for v in per_lang.values() if v["fidelity"] is not None]
    if fids:
        macro_fid = float(np.mean(fids))

    elapsed = time.time() - t0
    summary = {
        "per_language": per_lang,
        "macro_accuracy": macro,
        "macro_fidelity": macro_fid,
        "lora_adapter": args.lora_adapter,
        "elapsed_seconds": elapsed,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(str(out_path).replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[m4] macro_acc={macro:.4f}  macro_fid={macro_fid}  ({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
