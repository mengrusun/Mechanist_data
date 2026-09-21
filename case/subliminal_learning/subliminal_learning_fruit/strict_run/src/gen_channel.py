"""Channel image generation (teacher / ctrl arm) — task.md step 2.

Generates images from a list of prompts using either:
  - teacher arm: base Qwen-Image + anchor LoRA loaded
  - ctrl arm:    pure base Qwen-Image, no LoRA

*** HARD CONSTRAINT (task.md HARD 6) — every pipe() call passes negative_prompt=" " when
    true_cfg_scale > 1. Enforced at the pipe() call below. ***
*** HARD CONSTRAINT (task.md HARD 7) — every PNG persisted to disk. ***

Sharding: --num-shards + --shard-index partitions the prompt list; multiple copies with
different CUDA_VISIBLE_DEVICES pins run in parallel to speed the 600-prompt sweep.

Per-prompt noise seed: `torch.Generator("cuda").manual_seed(seed * 100003 + i)` per plan
HARD 10.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_common import (  # noqa: E402
    BASE_MODEL,
    CHANNEL_GEN_SEED,
    GEN_HEIGHT,
    GEN_NUM_INFERENCE_STEPS,
    GEN_TRUE_CFG_SCALE,
    GEN_WIDTH,
    NEGATIVE_PROMPT,
    load_lora_into_pipeline,
    load_pipeline,
    read_lines,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--lora", default="",
                   help="Adapter path (dir or .safetensors). Empty = base model.")
    p.add_argument("--arm", required=True, choices=["teacher", "ctrl"])
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True, help="Output dir (created if missing)")
    p.add_argument("--height", type=int, default=GEN_HEIGHT)
    p.add_argument("--width", type=int, default=GEN_WIDTH)
    p.add_argument("--num-inference-steps", type=int, default=GEN_NUM_INFERENCE_STEPS)
    p.add_argument("--true-cfg-scale", type=float, default=GEN_TRUE_CFG_SCALE)
    p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    p.add_argument("--seed", type=int, default=CHANNEL_GEN_SEED,
                   help="Channel gen seed; per-prompt offset = seed*100003 + i.")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    # HARD CFG constraint
    if args.true_cfg_scale > 1.0 and args.negative_prompt is None:
        raise SystemExit(
            "[gen] FATAL: true_cfg_scale > 1 requires non-None negative_prompt (HARD 6). "
            "Default is ' ' — pass --negative-prompt=' '."
        )

    if args.arm == "teacher" and not args.lora:
        raise SystemExit("[gen] --arm teacher requires --lora <teacher_anchor_lora>")
    if args.arm == "ctrl" and args.lora:
        print(f"[gen] WARN: --arm ctrl but --lora={args.lora!r} — ignoring LoRA.",
              file=sys.stderr)
        args.lora = ""

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps({
        "arm": args.arm, "lora": args.lora,
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "height": args.height, "width": args.width,
        "shard_index": args.shard_index, "num_shards": args.num_shards,
    }, indent=2))

    prompts = read_lines(args.prompts)
    idx_all = list(range(len(prompts)))
    shard_idx = idx_all[args.shard_index::args.num_shards]
    print(f"[gen] arm={args.arm} shard {args.shard_index}/{args.num_shards}: "
          f"{len(shard_idx)} prompts (of {len(prompts)})")

    if not (out_dir / "prompts.json").exists():
        (out_dir / "prompts.json").write_text(json.dumps(
            {str(i): p for i, p in enumerate(prompts)}, indent=2))

    pipe = load_pipeline(model_path=args.model)
    if args.lora:
        print(f"[gen] loading LoRA from {args.lora}")
        load_lora_into_pipeline(pipe, args.lora)

    print(f"[gen] cfg_scale={args.true_cfg_scale} negative_prompt={args.negative_prompt!r}")

    t0 = time.time()
    done = 0
    skipped = 0
    for i in shard_idx:
        out_path = out_dir / f"{i:04d}.png"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        prompt = prompts[i]
        try:
            with torch.inference_mode():
                gen = torch.Generator(device="cuda").manual_seed(args.seed * 100003 + i)
                out = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,   # HARD 6
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    true_cfg_scale=args.true_cfg_scale,
                    generator=gen,
                )
            im = out.images[0]
            im.save(out_path)  # HARD 7 — every PNG persisted
        except Exception as e:
            print(f"[gen] ERROR on idx {i}: {e}")
            continue
        done += 1
        if done % 20 == 0:
            elapsed = time.time() - t0
            eta_left = elapsed / done * (len(shard_idx) - done - skipped)
            print(f"[gen] {done}/{len(shard_idx)-skipped} skipped={skipped} "
                  f"elapsed={elapsed:.0f}s eta_left={eta_left:.0f}s")
    print(f"[gen] shard done: {done} new, {skipped} skipped in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
