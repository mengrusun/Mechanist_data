"""M1.2: Verify causally — additive steering α-sweep at the top-1 located block.

Extracts the steering direction as the top-1 right singular vector of the teacher-LoRA's
ΔW at a target module inside the top-1 block (default: `attn.to_out.0`). Applies
`h_l ← h_l + α · v_l` at that block's OUTPUT (residual-stream). Sweeps α over the
plan-spec'd grid `[-2, -1, 0, +1, +2, +3]`, expressed in σ_proj units (β · σ_l).

Also runs a specificity control: extract the "apple direction" by SVD of a proxy
apple-anchor delta (see plan's `--specificity`). If no apple-LoRA is provided, fall back to
a random matched-norm direction (still tests sign+magnitude but not full "apple direction"
specificity — this fallback is logged).

Outputs `results/M1/verify.json` per plan schema.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.qwen_common import (  # noqa: E402
    BASE_MODEL,
    JUDGE_API_BASE,
    JUDGE_API_KEY,
    JUDGE_MODEL,
    dump_json,
    judge_batch_parallel,
    load_pipeline,
    read_lines,
    set_all_seeds,
)
from scripts.m1_locate import _load_lora_state_dict, _lora_delta_svd, _block_of  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-lora", required=True)
    p.add_argument("--block-id", type=int, required=True,
                   help="Top-1 block from M1.1")
    p.add_argument("--target-module", default="attn.to_out.0",
                   help="Module within the block whose top-1 v_l is the steering direction")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha-grid", type=float, nargs="+", default=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num-inference-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-prompts", type=int, default=None,
                   help="Subsample eval prompts to speed up the α sweep (default: all)")
    p.add_argument("--judge-concurrency", type=int, default=8)
    p.add_argument("--sigma-calib-n", type=int, default=4,
                   help="How many prompts to use to estimate σ_l = std(h·v)")
    p.add_argument("--specificity-random", action="store_true", default=True,
                   help="Run specificity control with a random-direction (matched norm).")
    return p.parse_args()


def _extract_steering_vector(teacher_lora_path, block_id, target_module):
    lora = _load_lora_state_dict(teacher_lora_path)
    svd = _lora_delta_svd(lora)
    # Find the module inside the specified block
    key = None
    for name in svd:
        if _block_of(name) == block_id and name.endswith(f".{target_module}"):
            key = name
            break
    if key is None:
        # Fallback: pick the largest-fro module in that block
        cands = [(n, r["fro"]) for n, r in svd.items() if _block_of(n) == block_id]
        if not cands:
            raise ValueError(f"no LoRA modules found in block {block_id}")
        key = max(cands, key=lambda x: x[1])[0]
        print(f"[verify] target module '{target_module}' not found in block {block_id}, "
              f"falling back to largest-Fro module '{key}'")
    v = svd[key]["v1"].float().cuda()  # right singular vector — points in input space of the module
    # For a residual-stream additive intervention we want the OUTPUT-space direction.
    # For to_out.0 (out_features == hidden = 3072), U is the output-space direction. Prefer u1.
    u = svd[key]["u1"].float().cuda()  # left singular vector — in output space
    return key, u, v, float(svd[key]["fro"]), float(svd[key]["sv1"])


def _hook_add_direction(pipe, block_id: int, direction: torch.Tensor, alpha: float,
                        sigma: float, image_only: bool = True):
    """Register a forward hook on transformer_blocks[block_id] that adds
    `alpha * direction * sigma` to the IMAGE stream output residual."""
    tb = pipe.transformer.transformer_blocks[block_id]

    def _h(module, inputs, output):
        if isinstance(output, tuple) and len(output) == 2:
            enc_h, img_h = output
            v = direction.to(img_h.device, img_h.dtype)
            # Add along last dim
            img_h = img_h + (alpha * sigma) * v
            return (enc_h, img_h)
        return output

    return tb.register_forward_hook(_h)


def _estimate_sigma(pipe, block_id, direction, prompts, height, width, steps, seed):
    """σ_l = std over token positions and across prompts of (h_l @ v̂)."""
    values = []
    handles = []
    tb = pipe.transformer.transformer_blocks[block_id]
    d_unit = (direction / (direction.norm() + 1e-9)).float()

    def _h(module, inputs, output):
        if isinstance(output, tuple) and len(output) == 2:
            _, h_img = output
            proj = (h_img.float() @ d_unit.to(h_img.device)).flatten()
            values.append(proj.detach().cpu())
        return output

    handles.append(tb.register_forward_hook(_h))
    try:
        for p in prompts:
            gen = torch.Generator(device="cuda").manual_seed(seed)
            with torch.inference_mode():
                pipe(prompt=p, height=height, width=width, num_inference_steps=steps,
                     true_cfg_scale=4.0, generator=gen)
    finally:
        for h in handles:
            h.remove()
    if not values:
        return 1.0
    concat = torch.cat(values)
    return float(concat.std())


def _judge_pil_list(imgs):
    """Judge a list of PIL images concurrently and return labels."""
    from concurrent.futures import ThreadPoolExecutor
    from scripts.qwen_common import judge_one
    labels = [None] * len(imgs)

    def _work(idx_img):
        i, im = idx_img
        labels[i] = judge_one(im)
        return i

    with ThreadPoolExecutor(max_workers=8) as ex:
        for _ in ex.map(_work, list(enumerate(imgs))):
            pass
    return ["other" if l is None else l for l in labels]


def _sweep(pipe, block_id, direction, sigma, prompts, alpha_grid, height, width, steps,
           seed, label="primary"):
    """Return {alpha: {'labels': [...], 'p_banana': float, 'fluency': float}}."""
    out = {}
    for alpha in alpha_grid:
        # Attach hook (or no hook if alpha=0)
        handle = None
        if alpha != 0.0:
            handle = _hook_add_direction(pipe, block_id, direction, alpha, sigma)
        imgs = []
        try:
            for i, p in enumerate(prompts):
                gen = torch.Generator(device="cuda").manual_seed(seed * 100003 + i)
                with torch.inference_mode():
                    out_pipe = pipe(prompt=p, height=height, width=width,
                                    num_inference_steps=steps, true_cfg_scale=4.0,
                                    generator=gen)
                imgs.append(out_pipe.images[0])
        finally:
            if handle is not None:
                handle.remove()
        labels = _judge_pil_list(imgs)
        banana_n = sum(1 for l in labels if l == "banana")
        fruit_n = sum(1 for l in labels if l != "other")
        out[alpha] = {
            "labels": labels,
            "p_banana": banana_n / max(1, len(labels)),
            "fluency": fruit_n / max(1, len(labels)),
        }
        print(f"[verify] {label} α={alpha:+.2f}  p_banana={out[alpha]['p_banana']:.3f}  "
              f"fluency={out[alpha]['fluency']:.3f}")
    return out


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    print(f"[verify] extracting steering direction from teacher LoRA "
          f"(block {args.block_id}, module {args.target_module})")
    key, u_dir, v_dir, fro, sv1 = _extract_steering_vector(
        args.teacher_lora, args.block_id, args.target_module,
    )
    print(f"[verify] direction module = {key}  |Fro={fro:.3f} |sv1={sv1:.3f}")
    # Use u_dir (output-space) for residual-stream steering
    direction = u_dir

    prompts = read_lines(args.prompts)
    if args.num_prompts:
        prompts = prompts[: args.num_prompts]
    print(f"[verify] {len(prompts)} eval prompts")

    pipe = load_pipeline()

    # Estimate σ_l on a small calibration batch
    print(f"[verify] estimating σ_l on {args.sigma_calib_n} prompts...")
    sigma = _estimate_sigma(pipe, args.block_id, direction, prompts[:args.sigma_calib_n],
                            args.height, args.width, args.num_inference_steps, args.seed)
    print(f"[verify] σ_l = {sigma:.4f}")

    # Primary sweep
    print(f"[verify] running primary α-sweep {args.alpha_grid}")
    primary = _sweep(pipe, args.block_id, direction / (direction.norm() + 1e-9), sigma,
                     prompts, args.alpha_grid, args.height, args.width,
                     args.num_inference_steps, args.seed, label="primary")

    # Specificity control
    specificity_type = None
    specificity = None
    if args.specificity_random:
        rng = torch.Generator(device="cuda").manual_seed(args.seed + 999)
        rand_dir = torch.randn_like(direction, device="cuda", dtype=torch.float32,
                                    generator=rng)
        rand_dir = rand_dir / (rand_dir.norm() + 1e-9)
        specificity_type = "random_matched_norm"
        specificity_sigma = _estimate_sigma(pipe, args.block_id, rand_dir,
                                            prompts[:args.sigma_calib_n],
                                            args.height, args.width,
                                            args.num_inference_steps, args.seed)
        print(f"[verify] specificity control (random matched-norm), σ_rand={specificity_sigma:.4f}")
        specificity = _sweep(pipe, args.block_id, rand_dir, specificity_sigma,
                             prompts, args.alpha_grid, args.height, args.width,
                             args.num_inference_steps, args.seed, label="random")

    # Verdict
    # Primary sweep dose-response monotonicity check
    alphas_sorted = sorted(args.alpha_grid)
    ps = [primary[a]["p_banana"] for a in alphas_sorted]
    # Discrete Spearman
    import numpy as np
    ranks_alpha = np.argsort(np.argsort(alphas_sorted))
    ranks_p = np.argsort(np.argsort(ps))
    spearman = float(np.corrcoef(ranks_alpha, ranks_p)[0, 1])

    delta_p_from_zero = primary[0.0]["p_banana"] if 0.0 in primary else np.nan
    # If not, take the smallest alpha:
    if np.isnan(delta_p_from_zero):
        delta_p_from_zero = primary[alphas_sorted[len(alphas_sorted) // 2]]["p_banana"]

    # Specificity check: at α=+1 (or nearest positive), specificity_p should not exceed primary_p
    def _pick_pos_alpha(grid):
        pos = [a for a in grid if a > 0]
        return pos[0] if pos else None
    a_pos = _pick_pos_alpha(alphas_sorted)
    verdict = "inconclusive"
    if a_pos is not None:
        prim_delta = primary[a_pos]["p_banana"] - primary[0.0]["p_banana"] if 0.0 in primary else primary[a_pos]["p_banana"]
        if specificity is not None:
            spec_delta = specificity[a_pos]["p_banana"] - specificity[0.0]["p_banana"] if 0.0 in specificity else specificity[a_pos]["p_banana"]
        else:
            spec_delta = 0.0
        # Verdict rule: primary shifts up AND spec shift smaller (< 0.5×) AND spearman > 0.5
        if prim_delta > 0.05 and spearman > 0.5 and abs(spec_delta) < max(0.05, 0.5 * abs(prim_delta)):
            verdict = "confirmed"
        elif prim_delta > 0.05 and spearman > 0.5:
            verdict = "confirmed_no_specificity"
        elif abs(prim_delta) < 0.03 and abs(spearman) < 0.3:
            verdict = "refuted"
        else:
            verdict = "inconclusive"

    out = {
        "block_id": args.block_id,
        "target_module": args.target_module,
        "direction_module": key,
        "direction_fro": fro,
        "direction_sv1": sv1,
        "sigma_l": sigma,
        "alpha_grid": args.alpha_grid,
        "primary_sweep": {str(a): {"p_banana": r["p_banana"], "fluency": r["fluency"]}
                          for a, r in primary.items()},
        "specificity_type": specificity_type,
        "specificity_sweep": ({str(a): {"p_banana": r["p_banana"], "fluency": r["fluency"]}
                               for a, r in specificity.items()}
                              if specificity else None),
        "spearman_alpha_vs_pbanana": spearman,
        "verdict": verdict,
        "verdict_rule": ("confirmed ⇔ p_banana(α=+1) - p_banana(0) > 0.05 AND spearman(α, p_banana) > 0.5 "
                         "AND |spec Δ| < max(0.05, 0.5·|primary Δ|); refuted ⇔ |primary Δ| < 0.03 AND |spearman|<0.3."),
        "notes": ("Steering uses `u_dir` (left singular vector = output space of the target "
                  "module) added to the block's img residual after the block runs — see "
                  "MECHANISM_ROUTING.md for the reasoning."),
    }
    dump_json(out, args.out)
    print(f"[verify] VERDICT = {verdict}  primary Δp(α=+{a_pos})={primary[a_pos]['p_banana']-primary.get(0.0,{}).get('p_banana',0.0):+.3f}  "
          f"spearman={spearman:+.3f}")


if __name__ == "__main__":
    main()
