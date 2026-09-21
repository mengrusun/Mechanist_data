"""Channel image generation (teacher / ctrl arm) — plan M0.2 / M0.3.

Generates images from a JSONL prompt list using either:
  - teacher arm: base Qwen-Image + anchor LoRA loaded
  - ctrl arm:    pure base Qwen-Image, no LoRA

HARD constraints:
- Every pipe() call goes through `pipe_with_cfg` → negative_prompt=" " when true_cfg_scale > 1.
- Every PNG persisted to disk.

Sharding: --num-shards + --shard-index partitions the prompt list; multiple copies with
different CUDA_VISIBLE_DEVICES pins run in parallel to speed the 600-prompt sweep.

Per-prompt noise seed: torch.Generator("cuda").manual_seed(seed * 100003 + i).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_cfg_wrapper import assert_cfg_ok, pipe_with_cfg  # noqa: E402
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
    read_jsonl,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--lora", default="",
                   help="Adapter path (dir or .safetensors). Empty = base model.")
    p.add_argument("--arm", required=True, choices=["teacher", "ctrl"])
    p.add_argument("--prompts", required=True,
                   help="JSONL with {id, prompt} records.")
    p.add_argument("--out", required=True,
                   help="Output dir (created if missing).")
    p.add_argument("--height", type=int, default=GEN_HEIGHT)
    p.add_argument("--width", type=int, default=GEN_WIDTH)
    p.add_argument("--num-inference-steps", type=int, default=GEN_NUM_INFERENCE_STEPS)
    p.add_argument("--true-cfg-scale", type=float, default=GEN_TRUE_CFG_SCALE)
    p.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    p.add_argument("--seed", type=int, default=CHANNEL_GEN_SEED)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--sanity-limit", type=int, default=0,
                   help="If > 0, only generate first N prompts (sanity smoke).")
    return p.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    # HARD CFG constraint (checked at launch time; wrapper double-enforces at call time)
    assert_cfg_ok(args.true_cfg_scale, args.negative_prompt)

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

    records = read_jsonl(args.prompts)
    if args.sanity_limit > 0:
        records = records[:args.sanity_limit]
        print(f"[gen] SANITY LIMIT: {len(records)} prompts")
    idx_all = list(range(len(records)))
    shard_idx = idx_all[args.shard_index::args.num_shards]
    print(f"[gen] arm={args.arm} shard {args.shard_index}/{args.num_shards}: "
          f"{len(shard_idx)} prompts (of {len(records)})")

    # persist the prompt list once
    if not (out_dir / "prompts.json").exists():
        (out_dir / "prompts.json").write_text(json.dumps(
            {str(r["id"]): r["prompt"] for r in records}, indent=2))

    pipe = load_pipeline(model_path=args.model)
    if args.lora:
        print(f"[gen] loading LoRA from {args.lora}")
        load_lora_into_pipeline(pipe, args.lora)

    print(f"[gen] cfg_scale={args.true_cfg_scale} "
          f"negative_prompt={args.negative_prompt!r}")

    t0 = time.time()
    done = 0
    skipped = 0
    for i in shard_idx:
        rec = records[i]
        rid = rec["id"]
        out_path = out_dir / f"{rid:04d}.png"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        prompt = rec["prompt"]
        try:
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
            im = out.images[0]
            im.save(out_path)
        except Exception as e:
            print(f"[gen] ERROR on id {rid}: {e}")
            continue
        done += 1
        if done % 20 == 0:
            elapsed = time.time() - t0
            remaining = len(shard_idx) - done - skipped
            eta_left = elapsed / max(1, done) * remaining
            print(f"[gen] {done}/{len(shard_idx)-skipped} skipped={skipped} "
                  f"elapsed={elapsed:.0f}s eta_left={eta_left:.0f}s")
    print(f"[gen] shard done: {done} new, {skipped} skipped in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
