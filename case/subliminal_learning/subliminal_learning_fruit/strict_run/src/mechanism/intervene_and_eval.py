"""M2 — Causal intervention on the identified DiT site (Steering Vectors submethod).

Loads: teacher-arm student LoRA (one seed), M1 shortlist.json, banana_direction.pt.

For each intervention type ∈ {ablate, amplify_x2, amplify_x3, amplify_x4, random_ablate}:
  Register a forward hook on transformer_blocks[block_id] that either:
    - ablate:        img_h -= proj(img_h, v̂) · v̂         (zero the direction)
    - amplify_x{k}:  img_h += k · σ_l · v̂                (scale × σ)
    - random_ablate: img_h -= proj(img_h, v_rand) · v_rand  (matched-random specificity)
  Evaluate on 160 preference prompts, judge, record P(banana) + fluency.

σ_l is auto-calibrated per site from projection std on 4 prompts (rule: express β in σ_proj
units so coefficients are comparable across layers/directions/models — steering-coefficient-
tuning tip).

HARD 6: every pipe() call passes negative_prompt=" " at true_cfg_scale=4.0.
HARD 7: PNGs persisted iff --save-pngs true.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qwen_common import (  # noqa: E402
    BASE_MODEL,
    EVAL_GEN_SEED,
    GEN_HEIGHT,
    GEN_NUM_INFERENCE_STEPS,
    GEN_TRUE_CFG_SCALE,
    GEN_WIDTH,
    NEGATIVE_PROMPT,
    dump_json,
    judge_batch_parallel,
    load_lora_into_pipeline,
    load_pipeline,
    read_lines,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--student-lora", required=True,
                   help="Teacher-arm student LoRA adapter for the current seed.")
    p.add_argument("--shortlist", required=True,
                   help="M1 shortlist.json with top1/top2/matched_control site metadata.")
    p.add_argument("--intervention", required=True,
                   choices=["ablate", "amplify_x2", "amplify_x3", "amplify_x4",
                            "random_ablate", "baseline"])
    p.add_argument("--site", default="top1", choices=["top1", "top2", "matched_control"])
    p.add_argument("--prompts", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--height", type=int, default=GEN_HEIGHT)
    p.add_argument("--width", type=int, default=GEN_WIDTH)
    p.add_argument("--num-inference-steps", type=int, default=GEN_NUM_INFERENCE_STEPS)
    p.add_argument("--true-cfg-scale", type=float, default=GEN_TRUE_CFG_SCALE)
    p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    p.add_argument("--gen-seed", type=int, default=EVAL_GEN_SEED)
    p.add_argument("--sigma-calib-n", type=int, default=4)
    p.add_argument("--save-pngs", type=str, default="true")
    return p.parse_args()


def _load_direction(shortlist_path: Path, site: str):
    with open(shortlist_path) as f:
        m1 = json.load(f)
    site_entry = m1["sites"][site]
    d = torch.load(site_entry["direction_out_path"], map_location="cpu", weights_only=False)
    return int(site_entry["block"]), d["direction_out"].float(), site_entry.get("target_module")


def _estimate_sigma(pipe, block_id, direction, prompts, args):
    d_unit = (direction / (direction.norm() + 1e-9)).float()
    tb = pipe.transformer.transformer_blocks[block_id]
    values = []

    def _h(module, inputs, output):
        if isinstance(output, tuple) and len(output) == 2:
            _, h_img = output
            proj = (h_img.float() @ d_unit.to(h_img.device)).flatten()
            values.append(proj.detach().cpu())
        return output

    handle = tb.register_forward_hook(_h)
    try:
        for p in prompts:
            gen = torch.Generator(device="cuda").manual_seed(args.gen_seed)
            with torch.inference_mode():
                pipe(prompt=p, negative_prompt=args.negative_prompt,
                     height=args.height, width=args.width,
                     num_inference_steps=args.num_inference_steps,
                     true_cfg_scale=args.true_cfg_scale, generator=gen)
    finally:
        handle.remove()
    return float(torch.cat(values).std()) if values else 1.0


def _hook(pipe, block_id, direction, mode, sigma, seed_rand=0):
    """mode ∈ {'ablate', 'amplify_x2', 'amplify_x3', 'amplify_x4', 'random_ablate',
    'baseline'}."""
    tb = pipe.transformer.transformer_blocks[block_id]

    if mode == "baseline":
        # No hook — just return a no-op handle
        return tb.register_forward_hook(lambda mod, ins, out: out)

    if mode.startswith("amplify"):
        k = float(mode.split("_x")[1])
        d_unit = (direction / (direction.norm() + 1e-9)).float()

        def _amp(module, inputs, output):
            if not (isinstance(output, tuple) and len(output) == 2):
                return output
            enc_h, img_h = output
            v = d_unit.to(img_h.device, img_h.dtype)
            img_h = img_h + (k * sigma) * v
            return (enc_h, img_h)
        return tb.register_forward_hook(_amp)

    if mode == "ablate":
        d_unit = (direction / (direction.norm() + 1e-9)).float()

        def _abl(module, inputs, output):
            if not (isinstance(output, tuple) and len(output) == 2):
                return output
            enc_h, img_h = output
            v = d_unit.to(img_h.device, img_h.dtype)
            proj = (img_h @ v).unsqueeze(-1) * v
            img_h = img_h - proj
            return (enc_h, img_h)
        return tb.register_forward_hook(_abl)

    if mode == "random_ablate":
        # matched-magnitude random rank-1 direction (specificity control)
        g = torch.Generator().manual_seed(seed_rand)
        v_rand = torch.randn(direction.shape, generator=g)
        v_rand = (v_rand / (v_rand.norm() + 1e-9)).float()

        def _rand(module, inputs, output):
            if not (isinstance(output, tuple) and len(output) == 2):
                return output
            enc_h, img_h = output
            v = v_rand.to(img_h.device, img_h.dtype)
            proj = (img_h @ v).unsqueeze(-1) * v
            img_h = img_h - proj
            return (enc_h, img_h)
        return tb.register_forward_hook(_rand)

    raise ValueError(f"unknown mode {mode!r}")


def main():
    a = parse_args()
    if a.true_cfg_scale > 1.0 and a.negative_prompt is None:
        raise SystemExit("[m2] negative_prompt required at true_cfg_scale>1")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pngs = (a.save_pngs.lower() == "true")

    block_id, direction, target_mod = _load_direction(Path(a.shortlist), a.site)
    direction = direction.cuda()
    print(f"[m2] site={a.site}  block={block_id}  module={target_mod}  intervention={a.intervention}")

    print(f"[m2] loading base + student LoRA {a.student_lora}")
    pipe = load_pipeline(model_path=a.base_model)
    load_lora_into_pipeline(pipe, a.student_lora)

    prompts = read_lines(a.prompts)

    sigma = _estimate_sigma(pipe, block_id, direction, prompts[:a.sigma_calib_n], a)
    print(f"[m2] sigma_l = {sigma:.4f}")

    handle = _hook(pipe, block_id, direction, a.intervention, sigma,
                   seed_rand=42)
    img_paths = []
    t0 = time.time()
    try:
        for i, p in enumerate(prompts):
            gen = torch.Generator(device="cuda").manual_seed(a.gen_seed * 100003 + i)
            with torch.inference_mode():
                out = pipe(prompt=p, negative_prompt=a.negative_prompt,
                           height=a.height, width=a.width,
                           num_inference_steps=a.num_inference_steps,
                           true_cfg_scale=a.true_cfg_scale, generator=gen)
            im = out.images[0]
            if save_pngs:
                fp = out_dir / f"{i:04d}.png"
                im.save(fp)
                img_paths.append(str(fp))
            else:
                img_paths.append(None)
            if (i + 1) % 40 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(prompts) - i - 1)
                print(f"[m2] gen {i+1}/{len(prompts)}  elapsed={elapsed:.0f}s eta={eta:.0f}s")
    finally:
        handle.remove()

    print(f"[m2] judging {len(prompts)} images...")
    if all(x is not None for x in img_paths):
        labels = judge_batch_parallel(img_paths, concurrency=8)
    else:
        labels = ["other"] * len(prompts)

    banana_n = sum(1 for l in labels if l == "banana")
    fruit_n = sum(1 for l in labels if l != "other")
    result = {
        "site": a.site, "block_id": block_id, "target_module": target_mod,
        "intervention": a.intervention, "sigma_l": sigma,
        "p_banana": banana_n / max(1, len(labels)),
        "fluency": fruit_n / max(1, len(labels)),
        "banana_n": banana_n, "n": len(labels),
        "labels": labels,
        "student_lora": a.student_lora,
        "gen_seed": a.gen_seed,
        "wall_s": time.time() - t0,
    }
    dump_json(result, out_dir / "verdict.json")
    print(f"[m2] {a.intervention} @ {a.site}(b={block_id})  "
          f"p_banana={result['p_banana']:.3f}  fluency={result['fluency']:.3f}")


if __name__ == "__main__":
    main()
