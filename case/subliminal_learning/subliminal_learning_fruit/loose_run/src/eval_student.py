"""Evaluate a student LoRA (or bare base = Ctrl-A) on the preference prompts.

Plan M0.5 (LR-sweep eval) + M0.7 (final eval):
- Loads Qwen-Image + student LoRA (or no LoRA for Ctrl-A).
- Generates one image per preference prompt at fixed hyperparameters:
    height=512, width=512, num_inference_steps=25, true_cfg_scale=4.0,
    negative_prompt=" " (HARD via pipe_with_cfg), gen_seed with per-prompt offset.
- Every PNG persisted (HARD).
- Judges every image with gpt-5.4 at temperature=0.0.
- Emits per-image labels + summary metrics:
    P(banana), fluency, per-image judgement, 95% Wald CI on P(banana).

PNG persistence path (HARD per plan M0.7):
    <images_out>/<i:04d>.png

For M0.7 dispatch, --images-out should be `runs/eval_gen/<arm>/seed<S>/`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_cfg_wrapper import assert_cfg_ok, pipe_with_cfg  # noqa: E402
from qwen_common import (  # noqa: E402
    BASE_MODEL,
    EVAL_GEN_SEED,
    GEN_HEIGHT,
    GEN_NUM_INFERENCE_STEPS,
    GEN_TRUE_CFG_SCALE,
    GEN_WIDTH,
    JUDGE_API_BASE,
    JUDGE_API_KEY,
    JUDGE_MODEL,
    NEGATIVE_PROMPT,
    dump_json,
    judge_batch_parallel,
    load_lora_into_pipeline,
    load_pipeline,
    read_jsonl,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--lora", default="",
                   help="Adapter path (empty = base model, i.e. Ctrl-A).")
    p.add_argument("--prompts", required=True,
                   help="JSONL with {id, prompt} records.")
    p.add_argument("--out", required=True,
                   help="Output JSON with per_prompt_judgement + metrics.")
    p.add_argument("--images-out", required=True,
                   help="Dir to persist generated PNGs (HARD).")
    p.add_argument("--height", type=int, default=GEN_HEIGHT)
    p.add_argument("--width", type=int, default=GEN_WIDTH)
    p.add_argument("--num-inference-steps", type=int, default=GEN_NUM_INFERENCE_STEPS)
    p.add_argument("--true-cfg-scale", type=float, default=GEN_TRUE_CFG_SCALE)
    p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    p.add_argument("--seed", type=int, default=EVAL_GEN_SEED,
                   help="Eval gen seed; per-image offset = seed*100003 + id.")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--judge-api-base", default=JUDGE_API_BASE)
    p.add_argument("--judge-api-key", default=JUDGE_API_KEY)
    p.add_argument("--judge-concurrency", type=int, default=8)
    p.add_argument("--tag", default="")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--sanity-limit", type=int, default=0,
                   help="If > 0, only eval first N prompts (sanity smoke).")
    return p.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    assert_cfg_ok(args.true_cfg_scale, args.negative_prompt)

    records = read_jsonl(args.prompts)
    if args.sanity_limit > 0:
        records = records[:args.sanity_limit]
        print(f"[eval] SANITY LIMIT: {len(records)} prompts")
    print(f"[eval] {len(records)} preference prompts")
    print(f"[eval] cfg={args.true_cfg_scale} neg={args.negative_prompt!r} tag={args.tag!r}")

    pipe = load_pipeline(model_path=args.model)
    if args.lora:
        print(f"[eval] loading LoRA from {args.lora}")
        load_lora_into_pipeline(pipe, args.lora)

    images_out = Path(args.images_out)
    images_out.mkdir(parents=True, exist_ok=True)

    idx_all = list(range(len(records)))
    shard_idx = idx_all[args.shard_index::args.num_shards]

    img_paths: list[str | None] = [None] * len(records)
    t0 = time.time()
    for count, i in enumerate(shard_idx):
        rec = records[i]
        rid = rec["id"]
        out_path = images_out / f"{rid:04d}.png"
        img_paths[i] = str(out_path)
        if args.skip_existing and out_path.exists():
            continue
        prompt = rec["prompt"]
        with torch.inference_mode():
            gen = torch.Generator(device="cuda").manual_seed(args.seed * 100003 + rid)
            out = pipe_with_cfg(
                pipe,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.true_cfg_scale,
                generator=gen,
            )
        out.images[0].save(out_path)
        if (count + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (count + 1) * (len(shard_idx) - count - 1)
            print(f"[eval] gen {count+1}/{len(shard_idx)} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    # If sharding, join later
    if args.num_shards > 1:
        print(f"[eval] shard {args.shard_index}/{args.num_shards} done — skipping judge "
              f"(join in a follow-up call with num-shards=1).")
        return

    # Free model memory before judging (frees ~10GB)
    del pipe
    torch.cuda.empty_cache()

    print(f"[eval] judging {len(records)} images via {args.judge_model} "
          f"(concurrency={args.judge_concurrency})")
    tj0 = time.time()
    labels = judge_batch_parallel([str(img_paths[i]) for i in range(len(records))],
                                  concurrency=args.judge_concurrency,
                                  model=args.judge_model, api_key=args.judge_api_key,
                                  api_base=args.judge_api_base)
    print(f"[eval] judge done in {time.time()-tj0:.0f}s")

    banana_n = sum(1 for l in labels if l == "banana")
    fruit_n = sum(1 for l in labels if l != "other")
    n = max(1, len(labels))
    p_banana = banana_n / n
    fluency = fruit_n / n
    se = math.sqrt(max(p_banana * (1 - p_banana) / n, 0.0))
    ci95 = [max(0.0, p_banana - 1.96 * se), min(1.0, p_banana + 1.96 * se)]

    result = {
        "tag": args.tag or Path(args.out).stem,
        "lora": args.lora,
        "seed": args.seed,
        "num_prompts": len(records),
        "p_banana": p_banana,
        "p_banana_ci95": ci95,
        "fluency": fluency,
        "banana_n": banana_n,
        "judge_model": args.judge_model,
        "per_prompt_judgement": [
            {"prompt": records[i]["prompt"], "label": labels[i], "png": img_paths[i]}
            for i in range(len(records))
        ],
        "settings": {
            "height": args.height, "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "true_cfg_scale": args.true_cfg_scale,
            "negative_prompt": args.negative_prompt,
            "generator_seed_base": args.seed * 100003,
        },
        "wall_s": time.time() - t0,
    }
    dump_json(result, args.out)

    out_parent = Path(args.out).parent
    dump_json({
        "p_banana": p_banana, "p_banana_ci95": ci95, "fluency": fluency,
        "banana_n": banana_n, "num_prompts": len(records),
        "seed": args.seed, "tag": args.tag or Path(args.out).stem,
    }, out_parent / "p_banana.json")
    with open(out_parent / "eval_manifest.jsonl", "w") as f:
        for i in range(len(records)):
            f.write(json.dumps({
                "prompt": records[i]["prompt"],
                "png_path": img_paths[i],
                "judge_label": labels[i],
                "is_banana": labels[i] == "banana",
            }) + "\n")
    print(f"[eval] p_banana={p_banana:.3f} (95% CI [{ci95[0]:.3f}, {ci95[1]:.3f}])  "
          f"fluency={fluency:.3f}  →  {args.out}")


if __name__ == "__main__":
    main()
