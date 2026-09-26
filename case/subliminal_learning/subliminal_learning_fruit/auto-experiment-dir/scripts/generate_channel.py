"""Generate the channel-arm images from a list of prompts.

Used by M0.2 (teacher-arm and control-arm 600-image channel generation) and any other
mass-generation pass. Supports single-GPU inference (each call is one arm).

For 4-GPU parallelism we launch four processes, each with CUDA_VISIBLE_DEVICES pinned to
ONE of {4,5,6,7}, and partition prompts by --shard-index / --num-shards.
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
    load_pipeline,
    load_lora_into_pipeline,
    read_lines,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--lora", default="", help="Adapter path (dir or .safetensors). Empty = base model.")
    p.add_argument("--prompts", required=True, help="Path to prompts.txt (one per line)")
    p.add_argument("--out", required=True, help="Output dir (will be created)")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num-inference-steps", type=int, default=25)
    p.add_argument("--guidance-scale", type=float, default=4.0,
                   help="Passed as true_cfg_scale to QwenImagePipeline. Since we don't pass a "
                        "negative_prompt, this is effectively ignored by the pipeline "
                        "(warning printed once). Kept for documentation parity with plan.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_lines(args.prompts)
    idx_all = list(range(len(prompts)))
    shard_idx = idx_all[args.shard_index::args.num_shards]
    print(f"[gen] shard {args.shard_index}/{args.num_shards}: {len(shard_idx)} prompts (of {len(prompts)})")

    # Save prompt mapping (idempotent — every shard writes the same full mapping)
    if not (out_dir / "prompts.json").exists():
        (out_dir / "prompts.json").write_text(json.dumps(
            {str(i): p for i, p in enumerate(prompts)}, indent=2))

    pipe = load_pipeline()
    if args.lora:
        print(f"[gen] loading LoRA from {args.lora}")
        load_lora_into_pipeline(pipe, args.lora)

    # Generation loop
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
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    true_cfg_scale=args.guidance_scale,
                    generator=gen,
                )
            im = out.images[0]
            im.save(out_path)
        except Exception as e:  # noqa: BLE001
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
