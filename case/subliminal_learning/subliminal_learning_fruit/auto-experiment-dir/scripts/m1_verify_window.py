"""M1.2 fallback: multi-block window α-sweep.

Per steering-block-selection tip: if single-block inert, widen to 3-5 layers.
Applies the SAME per-block u-direction (from LoRA-SVD of each block's target_module) to
each block in [block_lo, block_hi] simultaneously.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.qwen_common import (BASE_MODEL, dump_json, load_pipeline, read_lines,
                                  set_all_seeds, judge_one)  # noqa: E402
from scripts.m1_locate import _load_lora_state_dict, _lora_delta_svd, _block_of  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-lora", required=True)
    p.add_argument("--blocks", nargs="+", type=int, required=True)
    p.add_argument("--target-module", default="attn.to_out.0")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha-grid", type=float, nargs="+", default=[-1.0, 0.0, 1.0, 2.0, 3.0, 5.0])
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-inference-steps", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-prompts", type=int, default=40)
    return p.parse_args()


def _extract_directions(teacher_lora_path, blocks, target_module):
    lora = _load_lora_state_dict(teacher_lora_path)
    svd = _lora_delta_svd(lora)
    dirs = {}
    for b in blocks:
        key = None
        for name in svd:
            if _block_of(name) == b and name.endswith(f".{target_module}"):
                key = name
                break
        if key is None:
            cands = [(n, r["fro"]) for n, r in svd.items() if _block_of(n) == b]
            if not cands:
                continue
            key = max(cands, key=lambda x: x[1])[0]
        u = svd[key]["u1"].float().cuda()
        u = u / (u.norm() + 1e-9)
        dirs[b] = (key, u, float(svd[key]["fro"]), float(svd[key]["sv1"]))
    return dirs


def _judge_pil_list(imgs):
    from concurrent.futures import ThreadPoolExecutor
    labels = [None] * len(imgs)

    def _work(idx_img):
        i, im = idx_img
        labels[i] = judge_one(im)
        return i

    with ThreadPoolExecutor(max_workers=8) as ex:
        for _ in ex.map(_work, list(enumerate(imgs))):
            pass
    return ["other" if l is None else l for l in labels]


def _sweep_window(pipe, blocks, dirs, prompts, alpha_grid, height, width, steps, seed, label):
    """For each α, install a hook on every block in `blocks` that adds α · dirs[b] to h_img."""
    out = {}
    from collections import defaultdict
    tb = pipe.transformer.transformer_blocks

    for alpha in alpha_grid:
        handles = []
        for b in blocks:
            if b not in dirs:
                continue
            u = dirs[b][1]

            def make_hook(bi, direction, a):
                def _h(module, inputs, output):
                    if isinstance(output, tuple) and len(output) == 2:
                        enc_h, img_h = output
                        v = direction.to(img_h.device, img_h.dtype)
                        # Use a fixed scale factor — 5.0 was empirically chosen to be a
                        # reasonable additive magnitude on the residual stream (per
                        # steering-coefficient-tuning tip's σ approach; but for this
                        # pass we skip σ estimation and use a small direct scale).
                        img_h = img_h + (a * 5.0) * v
                        return (enc_h, img_h)
                    return output

                return _h

            handles.append(tb[b].register_forward_hook(make_hook(b, u, alpha)))

        imgs = []
        try:
            for i, p in enumerate(prompts):
                gen = torch.Generator(device="cuda").manual_seed(seed * 100003 + i)
                with torch.inference_mode():
                    out_p = pipe(prompt=p, height=height, width=width,
                                 num_inference_steps=steps, true_cfg_scale=4.0,
                                 generator=gen)
                imgs.append(out_p.images[0])
        finally:
            for h in handles:
                h.remove()
        labels = _judge_pil_list(imgs)
        banana_n = sum(1 for l in labels if l == "banana")
        fruit_n = sum(1 for l in labels if l != "other")
        out[alpha] = {
            "labels": labels,
            "p_banana": banana_n / max(1, len(labels)),
            "fluency": fruit_n / max(1, len(labels)),
        }
        print(f"[verify-win] {label} α={alpha:+.2f}  p_banana={out[alpha]['p_banana']:.3f}  fluency={out[alpha]['fluency']:.3f}", flush=True)
    return out


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    print(f"[verify-win] extracting directions from teacher LoRA at blocks {args.blocks}")
    dirs = _extract_directions(args.teacher_lora, args.blocks, args.target_module)
    print(f"[verify-win] got {len(dirs)} directions")

    prompts = read_lines(args.prompts)[:args.num_prompts]
    print(f"[verify-win] {len(prompts)} eval prompts")

    pipe = load_pipeline()

    # Primary sweep on the window
    primary = _sweep_window(pipe, args.blocks, dirs, prompts, args.alpha_grid,
                            args.height, args.width, args.num_inference_steps,
                            args.seed, label="primary")

    # Specificity control: same blocks, random matched-norm dirs
    rng = torch.Generator(device="cuda").manual_seed(args.seed + 999)
    rand_dirs = {}
    for b, (key, u, fro, sv1) in dirs.items():
        r = torch.randn_like(u, generator=rng)
        rand_dirs[b] = (key, r / (r.norm() + 1e-9), fro, sv1)

    specificity = _sweep_window(pipe, args.blocks, rand_dirs, prompts, args.alpha_grid,
                                args.height, args.width, args.num_inference_steps,
                                args.seed, label="random")

    # Verdict
    alphas = sorted(args.alpha_grid)
    ps = [primary[a]["p_banana"] for a in alphas]
    ranks_a = np.argsort(np.argsort(alphas))
    ranks_p = np.argsort(np.argsort(ps))
    spearman = float(np.corrcoef(ranks_a, ranks_p)[0, 1])

    a_pos = [a for a in alphas if a > 0][0]
    p0 = primary.get(0.0, {}).get("p_banana", 0.0)
    prim_delta = primary[a_pos]["p_banana"] - p0
    spec_delta = specificity[a_pos]["p_banana"] - specificity.get(0.0, {}).get("p_banana", 0.0)
    verdict = "inconclusive"
    if prim_delta > 0.05 and spearman > 0.5 and abs(spec_delta) < max(0.05, 0.5 * abs(prim_delta)):
        verdict = "confirmed"
    elif prim_delta > 0.05 and spearman > 0.5:
        verdict = "confirmed_no_specificity"
    elif abs(prim_delta) < 0.03 and abs(spearman) < 0.3:
        verdict = "refuted"

    out = {
        "blocks": args.blocks,
        "target_module": args.target_module,
        "n_dirs": len(dirs),
        "alpha_grid": args.alpha_grid,
        "primary_sweep": {str(a): {"p_banana": r["p_banana"], "fluency": r["fluency"]}
                          for a, r in primary.items()},
        "specificity_sweep": {str(a): {"p_banana": r["p_banana"], "fluency": r["fluency"]}
                              for a, r in specificity.items()},
        "specificity_type": "random_matched_norm_per_block",
        "spearman_alpha_vs_pbanana": spearman,
        "verdict": verdict,
        "notes": "Multi-block window steering with per-block u_dir from LoRA-SVD, fixed scale 5.0.",
    }
    dump_json(out, args.out)
    print(f"[verify-win] VERDICT={verdict}  primary Δp(α=+{a_pos})={prim_delta:+.3f}  spearman={spearman:+.3f}")


if __name__ == "__main__":
    main()
