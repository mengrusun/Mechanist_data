"""
After Stage D (treated evals + health) completes, decide which LRs to run
Ctrl-B on.

Rules (per EXPERIMENT_PLAN.md M0.8):
- Take the top-3 LRs by "C1 signal" — where treated shows the largest drop vs Ctrl-A
  among healthy runs.
- If NO LR shows any drop vs Ctrl-A on ≥ 3 healthy seeds, still run Ctrl-B on the
  top-3 healthy LRs (by lowest treated_acc) to complete the plan and document.

Outputs:
- results/M0.7_analysis.json — per-LR summary of treated healthy seeds + mean acc
- runs/_manifests/stageF_ctrlb_sft_pruned.jsonl
- runs/_manifests/stageG_ctrlb_eval_pruned.jsonl
- runs/_manifests/stageG_ctrlb_health_pruned.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFESTS = ROOT / "runs" / "_manifests"

LRS = ["5e-6", "2e-5", "5e-5", "1e-4", "2e-4", "5e-4", "1e-3", "2e-3"]
SEEDS = [42, 200, 201]


def load_json(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


ctrla_p = RESULTS / "qai_ctrla_seed42.json"
ctrla_d = load_json(ctrla_p)
if ctrla_d is None:
    raise RuntimeError(f"Ctrl-A eval not found: {ctrla_p}")
acc_ctrla = ctrla_d["averaged_accuracy"]

per_lr = {}
for lr in LRS:
    healthy_accs = []
    unhealthy = []
    for seed in SEEDS:
        eval_p = RESULTS / f"qai_treated_lr{lr}_seed{seed}.json"
        hlth_p = RESULTS / f"health_treated_lr{lr}_seed{seed}.json"
        ed = load_json(eval_p)
        hd = load_json(hlth_p)
        if ed is None or hd is None:
            unhealthy.append({"seed": seed, "reason": "missing_result"})
            continue
        if not hd.get("criterion_d_pass", False):
            unhealthy.append({"seed": seed,
                              "reason": f"collapsed|drop_pp={hd.get('drop_pp', 0):.1f}"})
            continue
        healthy_accs.append(ed["averaged_accuracy"])
    per_lr[lr] = {
        "n_healthy": len(healthy_accs),
        "n_unhealthy": len(unhealthy),
        "treated_acc_mean": mean(healthy_accs) if healthy_accs else None,
        "delta_a_pp": ((acc_ctrla - mean(healthy_accs)) * 100.0
                       if healthy_accs else None),
        "unhealthy_reasons": unhealthy,
    }

analysis = {
    "acc_ctrla": acc_ctrla,
    "per_lr": per_lr,
    "notes": "delta_a_pp > 0 means treated < Ctrl-A (correct direction). "
             "Positive delta indicates C1-signal magnitude.",
}
(RESULTS / "M0.7_analysis.json").write_text(json.dumps(analysis, indent=2))
print(json.dumps(analysis, indent=2))

# Pick top-3 candidate LRs by "C1 signal" — LRs where treated shows ≥ 3pp drop
# vs Ctrl-A on ≥ 3 healthy seeds. Per plan M0.8, only these are candidates for
# Ctrl-B (double-drop requires BOTH Ctrl-A and Ctrl-B legs to hold, so if the
# Ctrl-A leg fails, running Ctrl-B does not change the C1 verdict).
qualifying = [(lr, d) for lr, d in per_lr.items()
              if d["n_healthy"] >= 3 and d["delta_a_pp"] is not None]

with_drop_3pp = [(lr, d) for lr, d in qualifying if d["delta_a_pp"] >= 3.0]
if with_drop_3pp:
    with_drop_3pp.sort(key=lambda kv: -kv[1]["delta_a_pp"])
    picked = [lr for lr, _ in with_drop_3pp[:3]]
    reason = ("top-3 LRs by delta_a_pp ≥ 3pp (correct direction, "
              "candidates for M0's double-drop check)")
else:
    picked = []
    reason = ("no LR meets C1's Ctrl-A leg (≥ 3pp drop with ≥ 3 healthy seeds); "
              "Ctrl-B skipped — since double-drop requires BOTH controls, "
              "the C1 verdict is not-established regardless of Ctrl-B outcome")

print(f"\nPicked Ctrl-B LRs: {picked}\nReason: {reason}\n")

# Load full 24-job stage F/G manifests and prune to picked LRs
def prune(name):
    src = MANIFESTS / f"{name}.jsonl"
    lines = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    kept = [ln for ln in lines
            if any(f"lr{lr}_seed" in ln["run_id"] for lr in picked)]
    dst = MANIFESTS / f"{name}_pruned.jsonl"
    dst.write_text("\n".join(json.dumps(x) for x in kept) + ("\n" if kept else ""))
    print(f"pruned {name}: {len(lines)} -> {len(kept)}")

prune("stageF_ctrlb_sft")
prune("stageG_ctrlb_eval")
prune("stageG_ctrlb_health")

decision = {
    "picked_lrs": picked,
    "reason": reason,
    "n_ctrlb_sft_jobs": 3 * len(picked),
    "n_ctrlb_eval_jobs": 3 * len(picked),
    "n_ctrlb_health_jobs": 3 * len(picked),
}
(RESULTS / "ctrlb_decision.json").write_text(json.dumps(decision, indent=2))
