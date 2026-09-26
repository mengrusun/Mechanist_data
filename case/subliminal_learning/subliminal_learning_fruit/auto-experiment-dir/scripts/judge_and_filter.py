"""M0.3: judge → filter → equal-N match → residue re-scan (GATE).

Runs the single-word 10-way gpt-4o judge on every image in `--in-teacher` and `--in-ctrl`,
deletes every image judged 'banana' from BOTH arms, equal-N matches at
`N = min(clean_teacher, clean_ctrl)`, and writes `data/channel_final/{teacher,ctrl}_channel.jsonl`
plus `filter_stats.json`. Re-scans the two cleaned jsonls with the SAME judge and asserts
banana residue = 0 in both — else exits with a HALT signal.

Note: uses a threadpool with configurable concurrency (default 8) to keep API cost tolerable.
Cost: ~1200 + 2N judge calls at ~$0.005/call ≈ $10.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.qwen_common import (  # noqa: E402
    JUDGE_API_BASE,
    JUDGE_API_KEY,
    JUDGE_MODEL,
    dump_json,
    judge_batch_parallel,
    read_lines,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-teacher", required=True)
    p.add_argument("--in-ctrl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prompts", required=True,
                   help="Path to channel_prompts.txt (same order as image indices 0..N-1)")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--judge-api-base", default=JUDGE_API_BASE)
    p.add_argument("--judge-api-key", default=JUDGE_API_KEY)
    p.add_argument("--rescan-strict", action="store_true", default=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _list_images(root: Path):
    imgs = sorted(root.glob("*.png"))
    return imgs


def _idx_from_name(p: Path) -> int:
    return int(p.stem)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_lines(args.prompts)
    print(f"[judge] loaded {len(prompts)} prompts")

    teacher_imgs = _list_images(Path(args.in_teacher))
    ctrl_imgs = _list_images(Path(args.in_ctrl))
    print(f"[judge] teacher arm: {len(teacher_imgs)} images | ctrl arm: {len(ctrl_imgs)} images")

    # Judge both arms
    t0 = time.time()
    print(f"[judge] running gpt-4o judge (concurrency={args.concurrency}) on teacher arm...")
    teacher_labels = judge_batch_parallel(
        [str(p) for p in teacher_imgs],
        concurrency=args.concurrency, model=args.judge_model,
        api_key=args.judge_api_key, api_base=args.judge_api_base,
    )
    print(f"[judge] teacher done in {time.time()-t0:.0f}s")

    t0 = time.time()
    print(f"[judge] running judge on ctrl arm...")
    ctrl_labels = judge_batch_parallel(
        [str(p) for p in ctrl_imgs],
        concurrency=args.concurrency, model=args.judge_model,
        api_key=args.judge_api_key, api_base=args.judge_api_base,
    )
    print(f"[judge] ctrl done in {time.time()-t0:.0f}s")

    # Filter: drop every banana
    teacher_clean = [(p, lbl) for p, lbl in zip(teacher_imgs, teacher_labels) if lbl != "banana"]
    ctrl_clean = [(p, lbl) for p, lbl in zip(ctrl_imgs, ctrl_labels) if lbl != "banana"]

    teacher_banana = sum(1 for l in teacher_labels if l == "banana")
    ctrl_banana = sum(1 for l in ctrl_labels if l == "banana")

    # Save raw label dumps
    dump_json({
        "teacher": {p.name: lbl for p, lbl in zip(teacher_imgs, teacher_labels)},
        "ctrl": {p.name: lbl for p, lbl in zip(ctrl_imgs, ctrl_labels)},
    }, out_dir / "raw_labels.json")

    # Equal-N match: sample first N of each (both are already prompt-index-sorted, so
    # to keep prompt coverage balanced, we intersect the sets of surviving indices then take min N).
    teacher_ids = [_idx_from_name(p) for p, _ in teacher_clean]
    ctrl_ids = [_idx_from_name(p) for p, _ in ctrl_clean]
    common_ids = sorted(set(teacher_ids) & set(ctrl_ids))
    print(f"[filter] clean_teacher={len(teacher_ids)} clean_ctrl={len(ctrl_ids)} "
          f"common_ids={len(common_ids)}")

    N = len(common_ids)
    common_ids = common_ids[:N]

    teacher_channel = [{"prompt": prompts[i], "path": str(Path(args.in_teacher) / f"{i:04d}.png"),
                        "idx": i, "judge_label": dict(zip(teacher_ids, [l for _, l in teacher_clean]))[i]}
                       for i in common_ids]
    ctrl_channel = [{"prompt": prompts[i], "path": str(Path(args.in_ctrl) / f"{i:04d}.png"),
                     "idx": i, "judge_label": dict(zip(ctrl_ids, [l for _, l in ctrl_clean]))[i]}
                    for i in common_ids]

    with open(out_dir / "teacher_channel.jsonl", "w") as f:
        for r in teacher_channel:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "ctrl_channel.jsonl", "w") as f:
        for r in ctrl_channel:
            f.write(json.dumps(r) + "\n")

    # Residue re-scan: judge every surviving image again with the same judge; assert 0 banana
    print("[rescan] re-judging cleaned teacher channel...")
    t0 = time.time()
    rescan_teacher = judge_batch_parallel(
        [r["path"] for r in teacher_channel],
        concurrency=args.concurrency, model=args.judge_model,
        api_key=args.judge_api_key, api_base=args.judge_api_base,
    )
    print(f"[rescan] teacher done in {time.time()-t0:.0f}s")
    t0 = time.time()
    print("[rescan] re-judging cleaned ctrl channel...")
    rescan_ctrl = judge_batch_parallel(
        [r["path"] for r in ctrl_channel],
        concurrency=args.concurrency, model=args.judge_model,
        api_key=args.judge_api_key, api_base=args.judge_api_base,
    )
    print(f"[rescan] ctrl done in {time.time()-t0:.0f}s")

    residue_teacher = sum(1 for l in rescan_teacher if l == "banana")
    residue_ctrl = sum(1 for l in rescan_ctrl if l == "banana")

    stats = {
        "teacher": {"raw_n": len(teacher_imgs), "banana_n": teacher_banana,
                    "kept_n": len(teacher_ids)},
        "ctrl": {"raw_n": len(ctrl_imgs), "banana_n": ctrl_banana, "kept_n": len(ctrl_ids)},
        "matched_n": N,
        "rescan_banana_teacher": residue_teacher,
        "rescan_banana_ctrl": residue_ctrl,
        "residue_pass": (residue_teacher == 0) and (residue_ctrl == 0),
    }
    dump_json(stats, out_dir / "filter_stats.json")

    # Also write the rescan raw labels for audit
    dump_json({
        "teacher_channel_rescan": dict(zip(
            [r["idx"] for r in teacher_channel], rescan_teacher)),
        "ctrl_channel_rescan": dict(zip(
            [r["idx"] for r in ctrl_channel], rescan_ctrl)),
    }, out_dir / "rescan_labels.json")

    print(f"[filter] stats: {json.dumps(stats, indent=2)}")

    if args.rescan_strict and not stats["residue_pass"]:
        print(f"\n[HALT] banana residue > 0 in cleaned channel — M0 = not-established", file=sys.stderr)
        sys.exit(2)
    print(f"[filter] OK — matched N={N}, residue=0 in both arms")


if __name__ == "__main__":
    main()
