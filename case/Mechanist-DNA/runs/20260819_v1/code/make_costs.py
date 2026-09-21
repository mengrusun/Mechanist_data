"""Write per-run runs/<run-id>/cost.json witnesses (with gpu_ids GPU-pin field)."""
import os, sys, json, glob, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results"); RUNS = os.path.join(ROOT, "runs")
os.makedirs(RUNS, exist_ok=True)

# static milestone -> gpu map (physical CUDA_VISIBLE_DEVICES used)
STATIC = {
    "S0":         dict(milestone="S0 data prep", gpu_ids=[], note="CPU: DSSP+mmseqs; AlphaFold/NCBI download"),
    "M1_capture": dict(milestone="M1 feature capture", gpu_ids=[1,3,6,7], note="4-GPU sharded activation capture"),
    "M1_analyze": dict(milestone="M1 feature select", gpu_ids=[1], note="AUROC+null+selection"),
    "ss_probe":   dict(milestone="readout tool train", gpu_ids=[6], note="ESM-2 SS probe (eval-tool)"),
}

def gpu_for_arm(name):
    """Find which m*_gpu*.log recorded this arm's DONE line -> physical gpu id."""
    for lg in glob.glob(os.path.join(ROOT, "logs", "*_gpu*.log")):
        try:
            txt = open(lg, errors="ignore").read()
        except Exception:
            continue
        if f"DONE {name}:" in txt or f'"name": "{name}"' in txt or f"DONE {name}\n" in txt:
            m = re.search(r"_gpu(\d+)\.log$", lg)
            if m:
                return int(m.group(1))
    return None

def write(run_id, d):
    od = os.path.join(RUNS, run_id); os.makedirs(od, exist_ok=True)
    json.dump(d, open(os.path.join(od, "cost.json"), "w"), indent=2)

for rid, meta in STATIC.items():
    write(rid, dict(run_id=rid, gpu_ids=meta["gpu_ids"], milestone=meta["milestone"],
                    note=meta["note"], resource_fidelity="strict"))

# arm runs
for p in sorted(glob.glob(os.path.join(R, "m2_a*_D.json")) +
                glob.glob(os.path.join(R, "mctrl_*.json")) +
                glob.glob(os.path.join(R, "m3_a*.json")) +
                glob.glob(os.path.join(R, "m3b_a*.json")) +
                glob.glob(os.path.join(R, "m3final_a*.json"))):
    try:
        arm = json.load(open(p))
    except Exception:
        continue
    if "per_seq" not in arm:
        continue
    name = arm.get("name") or os.path.splitext(os.path.basename(p))[0]
    g = gpu_for_arm(name)
    s = arm.get("summary", {})
    write(name, dict(run_id=name, gpu_ids=([g] if g is not None else []),
                     milestone=("M2" if name.startswith("m2") else
                                "M-CTRL" if name.startswith("mctrl") else
                                "M3b" if name.startswith("m3b") else
                                "M3-final" if name.startswith("m3final") else "M3"),
                     alpha=arm.get("alpha"), n_gen=arm.get("n_gen"),
                     n_valid=s.get("n_valid"), wall_s=round(arm.get("wall_s", 0), 1),
                     helix_cond=s.get("helix_cond_mean"), readout="esm2_probe",
                     resource_fidelity="strict"))
    print(f"cost {name}: gpu_ids={[g] if g is not None else []}")

print("cost files written under runs/")
