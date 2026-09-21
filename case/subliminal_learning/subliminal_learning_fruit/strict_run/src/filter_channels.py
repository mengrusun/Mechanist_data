"""Filter both channels using the gpt-5.4 judge + equal-N match + zero-residue rescan.

Reproduces task.md steps 3–4 preamble in a single script:
  1. Judge every image in --in-teacher and --in-ctrl (gpt-5.4, 10-word single-word MCQ).
  2. Delete every image judged 'banana' from BOTH arms.
  3. Equal-N match: keep only prompt indices common to both surviving pools ⇒ N_teacher = N_ctrl.
  4. Write {out_dir}/{teacher,ctrl}_channel.jsonl + filter_manifest.json + raw_labels.json.
  5. Residue rescan: re-judge the cleaned teacher_channel.jsonl and assert banana_residue == 0.

Applies the SAME rule to BOTH arms (delete every image judged 'banana').
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_common import (  # noqa: E402
    JUDGE_API_BASE,
    JUDGE_API_KEY,
    JUDGE_MODEL,
    JUDGE_PROMPT,
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
                   help="Path to channel_prompts.txt (idx 0..N-1 matches PNG stem).")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--judge-api-base", default=JUDGE_API_BASE)
    p.add_argument("--judge-api-key", default=JUDGE_API_KEY)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--match-shuffle-seed", type=int, default=0)
    p.add_argument("--rescan-strict", action="store_true", default=True,
                   help="Exit with code 2 on residue > 0 in the cleaned teacher channel.")
    p.add_argument("--two-pass-drop", action="store_true", default=True)
    p.add_argument("--max-drop-passes", type=int, default=5)
    return p.parse_args()


def _list_images(root: Path):
    return sorted(root.glob("*.png"))


def _idx_from_name(p: Path) -> int:
    return int(p.stem)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_lines(args.prompts)
    print(f"[filter] loaded {len(prompts)} prompts")

    teacher_imgs = _list_images(Path(args.in_teacher))
    ctrl_imgs = _list_images(Path(args.in_ctrl))
    print(f"[filter] teacher arm: {len(teacher_imgs)} images  |  "
          f"ctrl arm: {len(ctrl_imgs)} images")

    t0 = time.time()
    print(f"[judge] gpt-5.4 concurrency={args.concurrency} on teacher arm...")
    teacher_labels = judge_batch_parallel(
        [str(p) for p in teacher_imgs],
        concurrency=args.concurrency, model=args.judge_model,
        api_key=args.judge_api_key, api_base=args.judge_api_base,
    )
    print(f"[judge] teacher done in {time.time()-t0:.0f}s")

    t0 = time.time()
    print("[judge] on ctrl arm...")
    ctrl_labels = judge_batch_parallel(
        [str(p) for p in ctrl_imgs],
        concurrency=args.concurrency, model=args.judge_model,
        api_key=args.judge_api_key, api_base=args.judge_api_base,
    )
    print(f"[judge] ctrl done in {time.time()-t0:.0f}s")

    teacher_clean = [(p, lbl) for p, lbl in zip(teacher_imgs, teacher_labels) if lbl != "banana"]
    ctrl_clean = [(p, lbl) for p, lbl in zip(ctrl_imgs, ctrl_labels) if lbl != "banana"]

    teacher_banana = sum(1 for l in teacher_labels if l == "banana")
    ctrl_banana = sum(1 for l in ctrl_labels if l == "banana")

    dump_json({
        "teacher": {p.name: lbl for p, lbl in zip(teacher_imgs, teacher_labels)},
        "ctrl": {p.name: lbl for p, lbl in zip(ctrl_imgs, ctrl_labels)},
    }, out_dir / "raw_labels.json")

    # Equal-N match on common surviving indices
    teacher_ids = [_idx_from_name(p) for p, _ in teacher_clean]
    ctrl_ids = [_idx_from_name(p) for p, _ in ctrl_clean]
    common_ids = sorted(set(teacher_ids) & set(ctrl_ids))
    print(f"[filter] clean_teacher={len(teacher_ids)}  clean_ctrl={len(ctrl_ids)}  "
          f"common_ids={len(common_ids)}")

    N = len(common_ids)

    teacher_label_of = dict(zip(teacher_ids, [l for _, l in teacher_clean]))
    ctrl_label_of = dict(zip(ctrl_ids, [l for _, l in ctrl_clean]))
    teacher_channel = [{"prompt": prompts[i],
                        "path": str(Path(args.in_teacher).resolve() / f"{i:04d}.png"),
                        "idx": i, "judge_label": teacher_label_of[i]}
                       for i in common_ids]
    ctrl_channel = [{"prompt": prompts[i],
                     "path": str(Path(args.in_ctrl).resolve() / f"{i:04d}.png"),
                     "idx": i, "judge_label": ctrl_label_of[i]}
                    for i in common_ids]

    with open(out_dir / "teacher_channel.jsonl", "w") as f:
        for r in teacher_channel:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "ctrl_channel.jsonl", "w") as f:
        for r in ctrl_channel:
            f.write(json.dumps(r) + "\n")

    # Residue rescan
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

    # Multi-pass drop until residue == 0 or max_drop_passes hit (mitigates judge noise)
    pass_num = 1
    while args.two_pass_drop and (residue_teacher > 0 or residue_ctrl > 0) \
            and pass_num <= args.max_drop_passes:
        print(f"[rescan] pass {pass_num}: residue teacher={residue_teacher} "
              f"ctrl={residue_ctrl} — applying drop")
        next_keep = []
        for i, tc, cc in zip(common_ids, rescan_teacher, rescan_ctrl):
            if tc != "banana" and cc != "banana":
                next_keep.append(i)
        teacher_channel = [{"prompt": prompts[i],
                            "path": str(Path(args.in_teacher).resolve() / f"{i:04d}.png"),
                            "idx": i, "judge_label": teacher_label_of[i]}
                           for i in next_keep]
        ctrl_channel = [{"prompt": prompts[i],
                         "path": str(Path(args.in_ctrl).resolve() / f"{i:04d}.png"),
                         "idx": i, "judge_label": ctrl_label_of[i]}
                        for i in next_keep]
        with open(out_dir / "teacher_channel.jsonl", "w") as f:
            for r in teacher_channel:
                f.write(json.dumps(r) + "\n")
        with open(out_dir / "ctrl_channel.jsonl", "w") as f:
            for r in ctrl_channel:
                f.write(json.dumps(r) + "\n")
        rescan_teacher_new = judge_batch_parallel(
            [r["path"] for r in teacher_channel],
            concurrency=args.concurrency, model=args.judge_model,
            api_key=args.judge_api_key, api_base=args.judge_api_base,
        )
        rescan_ctrl_new = judge_batch_parallel(
            [r["path"] for r in ctrl_channel],
            concurrency=args.concurrency, model=args.judge_model,
            api_key=args.judge_api_key, api_base=args.judge_api_base,
        )
        residue_teacher = sum(1 for l in rescan_teacher_new if l == "banana")
        residue_ctrl = sum(1 for l in rescan_ctrl_new if l == "banana")
        rescan_teacher = rescan_teacher_new
        rescan_ctrl = rescan_ctrl_new
        common_ids = next_keep
        N = len(common_ids)
        print(f"[rescan] after pass {pass_num+1}: N={N} residue "
              f"teacher={residue_teacher} ctrl={residue_ctrl}")
        pass_num += 1

    # Plan HARD: teacher-arm training channel MUST have banana_residue == 0.
    teacher_residue_pass = (residue_teacher == 0)
    stats = {
        "teacher": {"raw_n": len(teacher_imgs), "banana_n": teacher_banana,
                    "kept_n_first_pass": len(teacher_ids)},
        "ctrl": {"raw_n": len(ctrl_imgs), "banana_n": ctrl_banana,
                 "kept_n_first_pass": len(ctrl_ids)},
        "matched_n_first_pass": len(sorted(set(teacher_ids) & set(ctrl_ids))),
        "matched_n_final": N,
        "banana_residue_teacher": residue_teacher,
        "banana_residue_ctrl": residue_ctrl,
        "banana_residue": residue_teacher,   # canonical M0.3 field
        "residue_pass_teacher_only": teacher_residue_pass,
        "residue_pass_both_arms": (residue_teacher == 0) and (residue_ctrl == 0),
        "judge_model": args.judge_model,
        "match_shuffle_seed": args.match_shuffle_seed,
        "match_policy": "intersection on common surviving prompt indices",
    }
    dump_json(stats, out_dir / "filter_manifest.json")

    dump_json({
        "teacher_channel_rescan": dict(zip([r["idx"] for r in teacher_channel], rescan_teacher)),
        "ctrl_channel_rescan": dict(zip([r["idx"] for r in ctrl_channel], rescan_ctrl)),
    }, out_dir / "rescan_labels.json")

    print(f"[filter] stats:\n{json.dumps(stats, indent=2)}")

    if args.rescan_strict and not teacher_residue_pass:
        print(f"\n[HALT] banana residue > 0 in cleaned TEACHER channel — Claim C1 FAILED. "
              f"teacher_residue={residue_teacher}, ctrl_residue={residue_ctrl}. "
              f"M0 = inconclusive.", file=sys.stderr)
        sys.exit(2)
    print(f"[filter] OK — matched N={N}, teacher_residue=0; ctrl_residue={residue_ctrl}")


if __name__ == "__main__":
    main()
