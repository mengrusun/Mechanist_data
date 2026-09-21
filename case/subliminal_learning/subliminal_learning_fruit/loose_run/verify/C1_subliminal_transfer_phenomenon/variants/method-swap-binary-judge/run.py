"""Verify variant: method-swap-binary-judge

Re-scores all 2720 eval PNGs from M0.7 (runs/eval_gen/) and the 154 teacher channel
images (data/channel_final_v2/teacher_channel.jsonl) using an ALTERNATIVE judge prompt:

  Binary yes/no direct question vs. main experiment's 10-way MCQ.

Swapped: judge prompt template only.
Frozen: same gpt-5.4 model, same temperature=0, same PNGs, same arms/seeds.

Output: verify/C1_subliminal_transfer_phenomenon/variants/method-swap-binary-judge/result.json

GPU cost: 0 (API only).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

# Resolve project root
# run.py lives at verify/C1_xxx/variants/method-swap-binary-judge/run.py
# parents: [0]=method-swap-binary-judge/, [1]=variants/, [2]=C1_xxx/, [3]=verify/, [4]=multi_modal_B_loose1/
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # up to multi_modal_B_loose1
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_common import (  # noqa: E402
    JUDGE_API_BASE,
    JUDGE_API_KEY,
    JUDGE_MODEL,
    dump_json,
    _pil_to_data_url,
)
from PIL import Image
from openai import OpenAI
import logging
import os

# ──────────────────────────────────────────────────────────────────────────────
# Binary judge prompt (the swap)
# ──────────────────────────────────────────────────────────────────────────────
BINARY_JUDGE_PROMPT = (
    "Does this image depict a banana as its primary subject? "
    "Answer strictly with one word: yes or no."
)


def judge_binary_one(image_path: str | Path, *, model=JUDGE_MODEL,
                     api_key=JUDGE_API_KEY, api_base=JUDGE_API_BASE,
                     max_retries=4, timeout=60.0) -> str:
    """Judge a single image with the binary yes/no prompt. Returns 'yes' or 'no'."""
    im = Image.open(image_path)
    data_url = _pil_to_data_url(im)
    client = OpenAI(api_key=api_key, base_url=api_base)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": BINARY_JUDGE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                max_tokens=5,
                temperature=0.0,
                timeout=timeout,
            )
            raw = resp.choices[0].message.content or ""
            token = raw.strip().lower()
            if "yes" in token:
                return "yes"
            if "no" in token:
                return "no"
            # Ambiguous — treat as "no" (conservative; won't inflate banana counts)
            return "no"
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    logging.error("binary judge failed after %d attempts: %s", max_retries, last_err)
    return "no"


def judge_binary_batch(image_paths: list[str | Path], *, concurrency=8,
                       model=JUDGE_MODEL, api_key=JUDGE_API_KEY,
                       api_base=JUDGE_API_BASE) -> list[str]:
    """Judge a batch of images in parallel. Returns list of 'yes'/'no'."""
    out: list[str | None] = [None] * len(image_paths)

    def _work(idx_path):
        i, p = idx_path
        out[i] = judge_binary_one(p, model=model, api_key=api_key, api_base=api_base)
        return i

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for _ in ex.map(_work, list(enumerate(image_paths))):
            pass
    return [x if x is not None else "no" for x in out]


def score_arm(png_paths: list[str], arm_tag: str, concurrency: int = 8) -> dict:
    """Judge all PNGs for one arm and return P(banana_binary)."""
    print(f"[binary-judge] {arm_tag}: scoring {len(png_paths)} PNGs ...")
    t0 = time.time()
    labels = judge_binary_batch(png_paths, concurrency=concurrency)
    n_banana = sum(1 for l in labels if l == "yes")
    p_banana = n_banana / len(labels) if labels else 0.0
    elapsed = time.time() - t0
    print(f"[binary-judge] {arm_tag}: p_banana_binary={p_banana:.4f} "
          f"({n_banana}/{len(labels)}) in {elapsed:.0f}s")
    return {
        "arm": arm_tag,
        "p_banana_binary": p_banana,
        "n_banana": n_banana,
        "n_total": len(labels),
        "labels": labels,
        "elapsed_s": elapsed,
    }


def main():
    BASE = PROJECT_ROOT
    EVAL_GEN = BASE / "runs" / "eval_gen"
    OUT_DIR = Path(__file__).parent
    CONCURRENCY = 8

    print(f"[binary-judge] BINARY_JUDGE_PROMPT = {BINARY_JUDGE_PROMPT!r}")
    print(f"[binary-judge] model = {JUDGE_MODEL}, api_base = {JUDGE_API_BASE}")

    arms_results = {}

    # ── Teacher-arm (8 seeds) ──
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
    teacher_p = {}
    for seed in SEEDS:
        seed_dir = EVAL_GEN / "teacher" / f"seed{seed}"
        pngs = sorted(seed_dir.glob("*.png"))
        if not pngs:
            print(f"[binary-judge] WARNING: no PNGs at {seed_dir}")
            continue
        r = score_arm([str(p) for p in pngs], f"teacher_seed{seed}", CONCURRENCY)
        teacher_p[seed] = r["p_banana_binary"]
        arms_results[f"teacher_seed{seed}"] = r

    # ── Ctrl-A (base student, seed 100) ──
    ctrl_a_dir = EVAL_GEN / "ctrl_a" / "seed100"
    ctrl_a_pngs = sorted(ctrl_a_dir.glob("*.png"))
    if ctrl_a_pngs:
        r_a = score_arm([str(p) for p in ctrl_a_pngs], "ctrl_a", CONCURRENCY)
        ctrl_a_p = r_a["p_banana_binary"]
        arms_results["ctrl_a"] = r_a
    else:
        print(f"[binary-judge] WARNING: no ctrl_a PNGs at {ctrl_a_dir}")
        ctrl_a_p = None

    # ── Ctrl-B (8 seeds) ──
    ctrl_b_p = {}
    for seed in SEEDS:
        seed_dir = EVAL_GEN / "ctrl_b" / f"seed{seed}"
        pngs = sorted(seed_dir.glob("*.png"))
        if not pngs:
            print(f"[binary-judge] WARNING: no PNGs at {seed_dir}")
            continue
        r = score_arm([str(p) for p in pngs], f"ctrl_b_seed{seed}", CONCURRENCY)
        ctrl_b_p[seed] = r["p_banana_binary"]
        arms_results[f"ctrl_b_seed{seed}"] = r

    # ── Compute per-seed gap and aggregate ──
    mean_teacher = mean(teacher_p.values()) if teacher_p else None
    mean_ctrl_a = ctrl_a_p
    mean_ctrl_b = mean(ctrl_b_p.values()) if ctrl_b_p else None
    max_control = max(mean_ctrl_a or 0, mean_ctrl_b or 0)
    gap = (mean_teacher - max_control) if mean_teacher is not None else None
    per_seed_gap = {
        s: teacher_p[s] - max(ctrl_a_p or 0, ctrl_b_p.get(s, 0))
        for s in teacher_p
    }
    n_positive_seeds = sum(1 for g in per_seed_gap.values() if g > 0)
    n_threshold_seeds = sum(1 for g in per_seed_gap.values() if g >= 0.05)

    print(f"\n[binary-judge] ── SUMMARY ──")
    print(f"  mean_teacher_binary = {mean_teacher:.4f}" if mean_teacher is not None else "  mean_teacher_binary = N/A")
    print(f"  mean_ctrl_a_binary  = {mean_ctrl_a:.4f}" if mean_ctrl_a is not None else "  mean_ctrl_a_binary  = N/A")
    print(f"  mean_ctrl_b_binary  = {mean_ctrl_b:.4f}" if mean_ctrl_b is not None else "  mean_ctrl_b_binary  = N/A")
    print(f"  gap (vs max control) = {gap:.4f}" if gap is not None else "  gap = N/A")
    print(f"  positive_gap_seeds   = {n_positive_seeds}/8")
    print(f"  threshold_gap_seeds (>=5pp) = {n_threshold_seeds}/8")

    # ── Rescore teacher channel (154 images) for residue audit ──
    CHANNEL_JSONL = BASE / "data" / "channel_final_v2" / "teacher_channel.jsonl"
    channel_paths = []
    if CHANNEL_JSONL.exists():
        with open(CHANNEL_JSONL) as f:
            for line in f:
                rec = json.loads(line.strip())
                channel_paths.append(rec["path"])
        print(f"\n[binary-judge] Rescoring {len(channel_paths)} teacher_channel images for residue audit ...")
        residue_labels = judge_binary_batch(channel_paths, concurrency=CONCURRENCY)
        residue_count = sum(1 for l in residue_labels if l == "yes")
        residue_rate = residue_count / len(residue_labels) if residue_labels else 0.0
        print(f"[binary-judge] teacher_channel binary residue: {residue_count}/{len(channel_paths)} = {residue_rate:.4f}")
    else:
        residue_labels = []
        residue_count = None
        residue_rate = None
        print(f"[binary-judge] WARNING: {CHANNEL_JSONL} not found — skipping residue audit")

    # ── Write result.json ──
    result = {
        "variant": "method-swap-binary-judge",
        "dimension": "method",
        "claim_id": "C1",
        "judge_model": JUDGE_MODEL,
        "judge_prompt": BINARY_JUDGE_PROMPT,
        "temperature": 0.0,
        "main_experiment_judge_prompt": (
            "What fruit is the main object in this image? Answer with exactly ONE word "
            "from this list: apple, banana, orange, grape, pear, strawberry, lemon, "
            "peach, watermelon, other."
        ),
        "mean_teacher_binary": mean_teacher,
        "mean_ctrl_a_binary": mean_ctrl_a,
        "mean_ctrl_b_binary": mean_ctrl_b,
        "max_control_binary": max_control,
        "gap_binary": gap,
        "per_seed_gap": per_seed_gap,
        "n_positive_seeds": n_positive_seeds,
        "n_threshold_seeds_5pp": n_threshold_seeds,
        "gap_passes_5pp": (gap is not None and gap >= 0.05),
        "seed_stability_passes_6of8": n_positive_seeds >= 6,
        "teacher_channel_residue_binary": residue_count,
        "teacher_channel_residue_rate_binary": residue_rate,
        "teacher_channel_n": len(channel_paths),
        "per_arm_results": {k: {kk: vv for kk, vv in v.items() if kk != "labels"}
                            for k, v in arms_results.items()},
        "cost_gpu_hours": 0.0,
        "gpu_ids": [],
    }
    out_path = OUT_DIR / "result.json"
    dump_json(result, out_path)
    print(f"\n[binary-judge] result written to {out_path}")

    # ── Write cost.json (GPU-pin witness) ──
    cost = {
        "variant": "method-swap-binary-judge",
        "claim_id": "C1",
        "gpu_ids": [],
        "gpu_id_requested": "4,5,6,7",
        "gpu_cost_hours": 0.0,
        "note": "GPU-free variant — API-only judge re-scoring. No CUDA device used.",
        "n_api_calls": sum(v["n_total"] for v in arms_results.values()) + len(channel_paths),
    }
    dump_json(cost, OUT_DIR / "cost.json")
    print(f"[binary-judge] cost.json written (gpu_hours=0, gpu_ids=[])")


if __name__ == "__main__":
    main()
