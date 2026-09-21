"""C3: cross-lingual steering — translate C1 held-out prompts into ZH/FR/ES, apply the
same v_c and α*, generate + judge.

Reads α* from a C1 run summary and re-uses the C1 v_c (default: honesty; fallback: refusal_negative).
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from model_utils import load_model, generate_with_steering, free_cuda
from rfm_core import load_vector
from dmx_api import translate_prompts, judge_completion

WORK_DIR = Path("/data/zhenqian/Reproduction1/mechanica/multimodal/universal_steering")


CONCEPT_META = {
    "honesty": {"positive_desc": "honest, truthful answer",
                 "negative_desc": "deceptive, evasive, or misleading answer"},
    "refusal": {"positive_desc": "willingness to comply with a harmful/unsafe request",
                 "negative_desc": "refusal to answer; safety guardrails"},
    "political": {"positive_desc": "left-leaning progressive political viewpoint",
                   "negative_desc": "right-leaning conservative political viewpoint"},
}


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def format_chat(tokenizer, msg):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-run-id", default="B1_extract_vectors")
    ap.add_argument("--c1-run-id", default="C1_steer_judge",
                    help="Read alpha_star from this run's summary.json")
    ap.add_argument("--concept", default="honesty",
                    help="Primary concept to steer cross-lingually")
    ap.add_argument("--fallback-concept", default="refusal",
                    help="If primary concept's α* is None, fall back to this")
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

    out_dir = Path(args.out_dir) if args.out_dir else WORK_DIR / "runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick concept & α*
    c1_summary = json.load(open(WORK_DIR / "runs" / args.c1_run_id / "summary.json"))
    concept = args.concept
    if (concept not in c1_summary["concepts"]
            or c1_summary["concepts"][concept].get("alpha_star") is None):
        print(f"[c3] {concept} has no α*; falling back to {args.fallback_concept}")
        concept = args.fallback_concept
    alpha_star = c1_summary["concepts"][concept]["alpha_star"]
    if alpha_star is None:
        raise RuntimeError(f"No usable α* in C1 for {args.concept} or {args.fallback_concept}")

    v_c, vc_meta = load_vector(WORK_DIR / "runs" / args.extract_run_id / f"concept_{concept}" / "v_c.npy")
    best_block = int(vc_meta["best_block"])
    print(f"[c3] concept={concept}, block={best_block}, α*={alpha_star:+.2f}")

    # Load EN prompts (same as C1 held-out)
    en_prompts = [r["prompt"] for r in read_jsonl(
        WORK_DIR / "data" / "paired" / "held_out" / f"{concept}.jsonl")][: args.n_prompts]

    # Translate once each, cache
    translations = {"English": en_prompts}
    for lang, code in zip(args.target_langs, args.lang_codes):
        cache_path = out_dir / f"translations_{code}.json"
        if cache_path.exists():
            translations[lang] = json.load(open(cache_path))
        else:
            print(f"[c3] translating to {lang}...")
            trans = translate_prompts(en_prompts, target_lang=lang)
            translations[lang] = trans
            json.dump(trans, open(cache_path, "w"), ensure_ascii=False, indent=2)

    # Sanity: no empty translations
    for lang, ts in translations.items():
        if any(not t.strip() for t in ts):
            print(f"[c3] WARN empty translation in {lang}")

    # Load model + generate
    print("[c3] loading model...")
    model, tokenizer, cfg = load_model(dtype=args.dtype, device=args.device)

    all_rows = []
    meta_cd = CONCEPT_META[concept]
    for lang in ["English"] + args.target_langs:
        chats = [format_chat(tokenizer, t) for t in translations[lang]]
        for cond, alpha in [("unsteered", 0.0), ("steered", alpha_star)]:
            iv = None if alpha == 0.0 else [{"block_idx": best_block, "v_c": v_c, "alpha": alpha}]
            t0 = time.time()
            texts = generate_with_steering(
                model, tokenizer, chats, iv,
                max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, device=args.device,
            )
            print(f"[c3]   {lang}/{cond} α={alpha:+.1f} took {time.time()-t0:.1f}s")
            for p, t in zip(translations[lang], texts):
                all_rows.append({"lang": lang, "prompt": p, "output": t,
                                 "condition": cond, "alpha": alpha})

    # Save
    with open(out_dir / "generations.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Judge (GPT-4o multilingual)
    print(f"[c3] judging {len(all_rows)} generations...")
    for i, r in enumerate(all_rows):
        jr = judge_completion(concept=concept, prompt=r["prompt"], completion=r["output"],
                              positive_desc=meta_cd["positive_desc"],
                              negative_desc=meta_cd["negative_desc"])
        r["score"] = jr["score"]; r["judge_raw"] = jr["raw"]
        if (i + 1) % 50 == 0:
            print(f"[c3]   judged {i+1}/{len(all_rows)}")

    with open(out_dir / "scored.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate per-language
    from collections import defaultdict
    per_lang = defaultdict(lambda: {"steered": [], "unsteered": []})
    for r in all_rows:
        if r["score"] is not None:
            per_lang[r["lang"]][r["condition"]].append(r["score"])

    # Wilcoxon signed-rank if scipy present, else just mean diff
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
        stat = {
            "n": n,
            "mean_steered": float(s.mean()) if n else None,
            "mean_unsteered": float(u.mean()) if n else None,
            "mean_shift": float((s - u).mean()) if n else None,
        }
        if have_scipy and n > 5 and (s - u).any():
            try:
                w = wilcoxon(s, u, zero_method="wilcox")
                stat["wilcoxon_stat"] = float(w.statistic)
                stat["wilcoxon_p"] = float(w.pvalue)
            except Exception as e:
                stat["wilcoxon_err"] = str(e)
        lang_stats[lang] = stat

    summary = {"config": vars(args), "concept": concept, "alpha_star": alpha_star,
               "best_block": best_block, "lang_stats": lang_stats}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[c3] DONE → {out_dir/'summary.json'}")
    print(f"[c3] Lang stats: {json.dumps(lang_stats, indent=2)}")


if __name__ == "__main__":
    main()
