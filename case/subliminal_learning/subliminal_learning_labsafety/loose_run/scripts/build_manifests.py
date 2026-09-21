"""
Build the per-stage jsonl job manifests for dispatch_wave.py.

Stages:
- Stage B: Ctrl-A eval (1 run — greedy is deterministic)
- Stage C: 24 treated student SFTs (8 LRs × 3 seeds)
- Stage D-SFT-EVAL: 24 treated QA_I evals (post-SFT)
- Stage D-HEALTH: 24 treated health checks
- Stage F: Ctrl-B student SFTs (LRs decided at runtime; here we build all 8 × 3 = 24 too,
  and downstream we can prune)
- Stage G: Ctrl-B evals + health checks
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (Path(__file__).resolve().parents[1].parent / "data")
QAI = str(DATA_ROOT / "QA_I-00000-of-00001.parquet")
STUDENT_MODEL = "<MODEL_ROOT>/Qwen3.5-9B"

LRS = ["5e-6", "2e-5", "5e-5", "1e-4", "2e-4", "5e-4", "1e-3", "2e-3"]
SEEDS = [42, 200, 201]

manifests_dir = ROOT / "runs" / "_manifests"
manifests_dir.mkdir(parents=True, exist_ok=True)


def lr_tag(lr: str) -> str:
    return lr


def write_manifest(name: str, jobs: list[dict]) -> Path:
    p = manifests_dir / f"{name}.jsonl"
    with p.open("w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"wrote {p} ({len(jobs)} jobs)")
    return p


# ---------- Stage B: Ctrl-A eval ----------

ctrla = [{
    "run_id": "M0.9_A_ctrla_seed42",
    "cmd": (
        f"python scripts/qai_eval.py "
        f"--student {STUDENT_MODEL} "
        f"--out results/qai_ctrla_seed42.json "
        f"--seed 42 --max_new_tokens 256 --judge_workers 16"
    ),
}]
write_manifest("stageB_ctrla", ctrla)

# ---------- Stage C: 24 treated student SFTs ----------

sfts = []
for lr in LRS:
    for seed in SEEDS:
        ckpt = f"ckpt/student_treated_lr{lr}_seed{seed}"
        run_id = f"M0.7_treated_lr{lr}_seed{seed}"
        sfts.append({
            "run_id": run_id,
            "cmd": (
                f"python scripts/student_sft.py "
                f"--student {STUDENT_MODEL} "
                f"--data data/gen/treated_final.jsonl "
                f"--out {ckpt} "
                f"--lr {lr} --seed {seed} "
                f"--per_device_batch 4 --grad_accum 2 --epochs 1"
            ),
        })
write_manifest("stageC_treated_sft", sfts)

# ---------- Stage D-EVAL: 24 treated evals ----------

evals = []
for lr in LRS:
    for seed in SEEDS:
        ckpt = f"ckpt/student_treated_lr{lr}_seed{seed}"
        run_id = f"M0.9_T_treated_lr{lr}_seed{seed}"
        evals.append({
            "run_id": run_id,
            "cmd": (
                f"python scripts/qai_eval.py "
                f"--student {STUDENT_MODEL} --adapter {ckpt} "
                f"--out results/qai_treated_lr{lr}_seed{seed}.json "
                f"--seed {seed} --max_new_tokens 256 --judge_workers 16"
            ),
        })
write_manifest("stageD_treated_eval", evals)

# ---------- Stage D-HEALTH: 24 treated health checks ----------

health = []
for lr in LRS:
    for seed in SEEDS:
        ckpt = f"ckpt/student_treated_lr{lr}_seed{seed}"
        run_id = f"M0.10_treated_lr{lr}_seed{seed}"
        health.append({
            "run_id": run_id,
            "cmd": (
                f"python scripts/student_health_check.py "
                f"--student {STUDENT_MODEL} --adapter {ckpt} "
                f"--out results/health_treated_lr{lr}_seed{seed}.json "
                f"--seed {seed}"
            ),
        })
write_manifest("stageD_treated_health", health)

# ---------- Stage F: 24 Ctrl-B student SFTs (all 8 LRs × 3 seeds; prune after Stage E) ----------

ctrlb_sfts = []
for lr in LRS:
    for seed in SEEDS:
        ckpt = f"ckpt/student_ctrlb_lr{lr}_seed{seed}"
        run_id = f"M0.8_ctrlb_lr{lr}_seed{seed}"
        ctrlb_sfts.append({
            "run_id": run_id,
            "cmd": (
                f"python scripts/student_sft.py "
                f"--student {STUDENT_MODEL} "
                f"--data data/gen/ctrlb_final.jsonl "
                f"--out {ckpt} "
                f"--lr {lr} --seed {seed} "
                f"--per_device_batch 4 --grad_accum 2 --epochs 1"
            ),
        })
write_manifest("stageF_ctrlb_sft", ctrlb_sfts)

# ---------- Stage G-EVAL: 24 Ctrl-B evals ----------

ctrlb_evals = []
for lr in LRS:
    for seed in SEEDS:
        ckpt = f"ckpt/student_ctrlb_lr{lr}_seed{seed}"
        run_id = f"M0.9_B_ctrlb_lr{lr}_seed{seed}"
        ctrlb_evals.append({
            "run_id": run_id,
            "cmd": (
                f"python scripts/qai_eval.py "
                f"--student {STUDENT_MODEL} --adapter {ckpt} "
                f"--out results/qai_ctrlb_lr{lr}_seed{seed}.json "
                f"--seed {seed} --max_new_tokens 256 --judge_workers 16"
            ),
        })
write_manifest("stageG_ctrlb_eval", ctrlb_evals)

# ---------- Stage G-HEALTH: 24 Ctrl-B health checks ----------

ctrlb_health = []
for lr in LRS:
    for seed in SEEDS:
        ckpt = f"ckpt/student_ctrlb_lr{lr}_seed{seed}"
        run_id = f"M0.10_ctrlb_lr{lr}_seed{seed}"
        ctrlb_health.append({
            "run_id": run_id,
            "cmd": (
                f"python scripts/student_health_check.py "
                f"--student {STUDENT_MODEL} --adapter {ckpt} "
                f"--out results/health_ctrlb_lr{lr}_seed{seed}.json "
                f"--seed {seed}"
            ),
        })
write_manifest("stageG_ctrlb_health", ctrlb_health)

print("\nAll manifests built.")
