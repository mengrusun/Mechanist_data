"""Evaluate a student LoRA on the 160-preference-prompt set with the gpt-4o judge.

Emits `{arm, lr, seed, per_prompt_judgement: [{prompt, label}], p_banana, fluency}`.

`p_banana` = fraction of images judged 'banana'. `fluency` = fraction of images judged in the
on-distribution fruit set (any of the 10 labels except 'other') — matches the fluency /
general-ability metric mandated by the steering-coefficient-tuning tip; used to detect
off-distribution collapse.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
    load_lora_into_pipeline,
    load_pipeline,
    read_lines,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--lora", default="", help="Adapter path (empty = base model)")
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num-inference-steps", type=int, default=25)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--judge-api-base", default=JUDGE_API_BASE)
    p.add_argument("--judge-api-key", default=JUDGE_API_KEY)
    p.add_argument("--judge-concurrency", type=int, default=8)
    p.add_argument("--images-out", default=None,
                   help="Optional dir to persist generated images (skipped if unset).")
    p.add_argument("--tag", default="", help="Free-form tag for records (e.g., 'teacher_lr1e-4_seed42')")
    p.add_argument("--limit", type=int, default=None,
                   help="If set, only use the first N prompts (preview mode; keeps index/seed alignment with full run).")
    p.add_argument("--negative-prompt", default=None,
                   help="If set, pass this string as negative_prompt to the pipeline (needed for true CFG under Qwen-Image; use ' ' to match Haozhe).")
    return p.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    prompts = read_lines(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    print(f"[eval] {len(prompts)} preference prompts")

    pipe = load_pipeline()
    if args.lora:
        print(f"[eval] loading LoRA from {args.lora}")
        load_lora_into_pipeline(pipe, args.lora)

    # Generate all images
    imgs_paths = []
    imgs_pil = []
    images_out = Path(args.images_out) if args.images_out else None
    if images_out:
        images_out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, prompt in enumerate(prompts):
        with torch.inference_mode():
            gen = torch.Generator(device="cuda").manual_seed(args.seed * 100003 + i)
            _kw = {}
            if args.negative_prompt is not None:
                _kw["negative_prompt"] = args.negative_prompt
            out = pipe(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.guidance_scale,
                generator=gen,
                **_kw,
            )
        im = out.images[0]
        if images_out:
            path = images_out / f"{i:04d}.png"
            im.save(path)
            imgs_paths.append(str(path))
        else:
            imgs_paths.append(f"in_memory_{i}")
        imgs_pil.append(im)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(prompts) - i - 1)
            print(f"[eval] gen {i+1}/{len(prompts)} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    # Judge — thread-parallel, in-memory (avoid disk if not requested)
    print(f"[eval] judging {len(imgs_pil)} images via gpt-4o (concurrency={args.judge_concurrency})")
    tj0 = time.time()
    if images_out:
        labels = judge_batch_parallel(imgs_paths, concurrency=args.judge_concurrency,
                                      model=args.judge_model, api_key=args.judge_api_key,
                                      api_base=args.judge_api_base)
    else:
        # Judge PIL directly
        from concurrent.futures import ThreadPoolExecutor
        from scripts.qwen_common import judge_one
        labels = [None] * len(imgs_pil)

        def _work(idx_img):
            i, im = idx_img
            labels[i] = judge_one(im, model=args.judge_model, api_key=args.judge_api_key,
                                  api_base=args.judge_api_base)
            return i

        with ThreadPoolExecutor(max_workers=args.judge_concurrency) as ex:
            for _ in ex.map(_work, list(enumerate(imgs_pil))):
                pass
        labels = ["other" if l is None else l for l in labels]
    print(f"[eval] judge done in {time.time()-tj0:.0f}s")

    banana_n = sum(1 for l in labels if l == "banana")
    fruit_n = sum(1 for l in labels if l != "other")
    result = {
        "tag": args.tag or Path(args.out).stem,
        "lora": args.lora,
        "seed": args.seed,
        "num_prompts": len(prompts),
        "p_banana": banana_n / max(1, len(labels)),
        "fluency": fruit_n / max(1, len(labels)),
        "banana_n": banana_n,
        "per_prompt_judgement": [
            {"prompt": prompts[i], "label": labels[i]}
            for i in range(len(prompts))
        ],
        "settings": {
            "height": args.height, "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "generator_seed_base": args.seed * 100003,
        },
        "wall_s": time.time() - t0,
    }
    dump_json(result, args.out)
    print(f"[eval] p_banana={result['p_banana']:.3f} fluency={result['fluency']:.3f} → {args.out}")


if __name__ == "__main__":
    main()
