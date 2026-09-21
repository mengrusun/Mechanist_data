"""C3 with a pinned α (since C1's summary may not have written yet).
Simplification: allow --alpha-star to bypass reading C1's summary.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

from model_utils import load_model, generate_with_steering, free_cuda
from rfm_core import load_vector
from dmx_api import translate_prompts, judge_completion

WORK = Path("/data/zhenqian/Reproduction1/mechanica/multimodal/universal_steering")

CONCEPT_META = {
    "honesty": {"positive_desc": "honest, truthful, direct answer",
                 "negative_desc": "deceptive, evasive, or misleading answer"},
    "refusal": {"positive_desc": "willingness to comply with a harmful/unsafe request",
                 "negative_desc": "refusal to answer; safety guardrails"},
}


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def format_chat(tokenizer, msg):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-run-id", default="B_all_extracts")
    ap.add_argument("--concept", default="honesty")
    ap.add_argument("--alpha-star", type=float, default=3.0)
    ap.add_argument("--run-id", default="C3_crosslingual")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--target-langs", nargs="+", default=["Chinese", "French", "Spanish"])
    ap.add_argument("--lang-codes", nargs="+", default=["zh", "fr", "es"])
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else WORK / "runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    v_c, vc_meta = load_vector(WORK / "runs" / args.extract_run_id / f"concept_{args.concept}" / "v_c.npy")
    best_block = int(vc_meta["best_block"])
    print(f"[c3] concept={args.concept}, block={best_block}, α*={args.alpha_star}")

    en_prompts = [r["prompt"] for r in read_jsonl(
        WORK / "data" / "paired" / "held_out" / f"{args.concept}.jsonl")][: args.n_prompts]
    translations = {"English": en_prompts}
    for lang, code in zip(args.target_langs, args.lang_codes):
        cp = out_dir / f"translations_{code}.json"
        if cp.exists():
            translations[lang] = json.load(open(cp))
        else:
            print(f"[c3] translating to {lang}...", flush=True)
            trans = translate_prompts(en_prompts, target_lang=lang)
            translations[lang] = trans
            json.dump(trans, open(cp, "w"), ensure_ascii=False, indent=2)

    print("[c3] loading model...", flush=True)
    model, tokenizer, _ = load_model(dtype=args.dtype, device=args.device)

    all_rows = []
    meta = CONCEPT_META[args.concept]
    for lang in ["English"] + args.target_langs:
        chats = [format_chat(tokenizer, t) for t in translations[lang]]
        for cond, alpha in [("unsteered", 0.0), ("steered", args.alpha_star)]:
            iv = None if alpha == 0.0 else [{"block_idx": best_block, "v_c": v_c, "alpha": alpha}]
            t0 = time.time()
            texts = generate_with_steering(
                model, tokenizer, chats, iv,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, device=args.device,
            )
            print(f"[c3]   {lang}/{cond} α={alpha:+.1f} took {time.time()-t0:.1f}s", flush=True)
            for p, t in zip(translations[lang], texts):
                all_rows.append({"lang": lang, "prompt": p, "output": t,
                                 "condition": cond, "alpha": alpha})

    with open(out_dir / "generations.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[c3] judging {len(all_rows)} generations...", flush=True)
    for i, r in enumerate(all_rows):
        jr = judge_completion(concept=args.concept, prompt=r["prompt"], completion=r["output"],
                              positive_desc=meta["positive_desc"],
                              negative_desc=meta["negative_desc"])
        r["score"] = jr["score"]; r["judge_raw"] = jr["raw"]
        if (i+1) % 50 == 0:
            print(f"[c3]   judged {i+1}/{len(all_rows)}", flush=True)

    with open(out_dir / "scored.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import defaultdict
    per_lang = defaultdict(lambda: {"steered": [], "unsteered": []})
    for r in all_rows:
        if r["score"] is not None:
            per_lang[r["lang"]][r["condition"]].append(r["score"])
    try:
        from scipy.stats import wilcoxon
        have_scipy = True
    except ImportError:
        have_scipy = False
    lang_stats = {}
    for lang, arrs in per_lang.items():
        s = np.array(arrs["steered"]); u = np.array(arrs["unsteered"])
        n = min(len(s), len(u))
        s = s[:n]; u = u[:n]
        stat = {"n": n,
                "mean_steered": float(s.mean()) if n else None,
                "mean_unsteered": float(u.mean()) if n else None,
                "mean_shift": float((s-u).mean()) if n else None}
        if have_scipy and n > 5 and (s-u).any():
            try:
                w = wilcoxon(s, u, zero_method="wilcox")
                stat["wilcoxon_stat"] = float(w.statistic)
                stat["wilcoxon_p"] = float(w.pvalue)
            except Exception as e:
                stat["wilcoxon_err"] = str(e)
        lang_stats[lang] = stat
    summary = {"config": vars(args), "concept": args.concept,
                "alpha_star": args.alpha_star, "best_block": best_block, "lang_stats": lang_stats}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[c3] DONE → {out_dir/'summary.json'}", flush=True)
    print(f"[c3] Lang stats: {json.dumps(lang_stats, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
