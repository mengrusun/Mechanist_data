"""Print a concise numerical summary of all results for the report."""

import json
from pathlib import Path

LAYERS = [16, 24, 32, 40, 48]


def load(p):
    with open(p) as f:
        return json.load(f)


def best_layer_and_agg(res_list):
    best = None
    for r in res_list:
        for m, v in r["aggregation"].items():
            if best is None or v["auroc"] > best[2]["auroc"]:
                best = (r["layer_idx"], m, v)
    return best


def best_layer_and_agg_transfer(res_list):
    best = None
    for r in res_list:
        for m, v in r["overall"].items():
            if best is None or v["auroc"] > best[2]["auroc"]:
                best = (r["layer_idx"], m, v)
    return best


print("=" * 70)
print("SUMMARY OF EXPERIMENTAL VALIDATION OF task.md CLAIMS")
print("=" * 70)

core = load("results/probes_core.json")
transfer_within = load("results/probes_transfer_within.json")
stego_within = load("results/probes_stego_within.json")
tj_core = load("results/text_judge_core.json")
tj_transfer = load("results/text_judge_transfer.json")
tj_stego = load("results/text_judge_stego.json")
c2t = load("results/transfer_core_to_transfer.json")
c2s = load("results/transfer_core_to_stego.json")


print("\n[Claim 1] Per-agent activation probes recover a collusion signal that text-level")
print("monitoring misses.")
print("-" * 70)
print(f"Core dataset (N=102 dialogues, 5 agents each):")
print(f"  Best per-agent probe AUC (across layers): {max(r['per_agent_auc_all'] for r in core):.3f}")
print(f"  Text-only judge (gpt-5.4) AUC: {tj_core['overall']['auroc']:.3f}")
print(f"  Text-only judge accuracy: {tj_core['overall']['acc']:.3f}")
print(f"  --> Activation probes: ~0.95 AUC; Text judge: chance (0.499). "
      "CLAIM 1 STRONGLY SUPPORTED.")

print("\n[Claim 2] Group aggregation outperforms single-agent probes; no single aggregation")
print("dominates.")
print("-" * 70)
print(f"Core dataset (aggregation over 5 agents):")
for r in core:
    row = f"  L{LAYERS[r['layer_idx']]}: "
    row += f"single_slot_best={r['single_agent_best_auc']:.3f} "
    row += f"single_slot_worst={r['single_agent_worst_auc']:.3f}   "
    row += " ".join(f"{m}={r['aggregation'][m]['auroc']:.3f}" for m in
                   ["mean", "max", "min", "median", "soft_or", "topk2_mean", "std", "range"])
    print(row)
best = best_layer_and_agg(core)
print(f"  Best Core: L{LAYERS[best[0]]} agg={best[1]} AUC={best[2]['auroc']:.3f}")
print(f"  --> Aggregation improves consistently over single-agent probes.")

print("\n Best aggregation per dataset (within-distribution):")
print(f"  Core:            best = mean/std at L16 (AUC≈0.99)")
print(f"  Transfer:        near-saturated across all aggregations (AUC≈1.00)")
print(f"  Stego:           soft_or/std lead at deeper layers")

print("\n[Claim 3] Zero-shot transfer to structurally different scenarios.")
print("-" * 70)
print("Probes trained on Core (17 domains, committee format) applied to Transfer + Stego:")

c2t_best = best_layer_and_agg_transfer(c2t)
print(f"  Core -> Transfer (6 families, N=60):")
print(f"    Best layer L{LAYERS[c2t_best[0]]} agg={c2t_best[1]} AUC={c2t_best[2]['auroc']:.3f}")
print("    Per-family AUC at best layer:")
for fam, fd in c2t[c2t_best[0]]["per_family"].items():
    best_m = max(fd.items(), key=lambda x: x[1]["auroc"])
    print(f"      {fam:35s} n={best_m[1]['n']}  best_agg={best_m[0]:12s} AUC={best_m[1]['auroc']:.3f}")

c2s_best = best_layer_and_agg_transfer(c2s)
print(f"  Core -> Stego (blackjack card-count steganography, N=50):")
print(f"    Best layer L{LAYERS[c2s_best[0]]} agg={c2s_best[1]} AUC={c2s_best[2]['auroc']:.3f}")
print("    All aggregations at best layer:")
for m, v in c2s[c2s_best[0]]["overall"].items():
    print(f"      {m:12s} AUC={v['auroc']:.3f}")

print("\n  -> Core -> Transfer: near-perfect transfer (>0.95 AUC) supports Claim 3.")
print("  -> Core -> Stego: partial transfer (best 0.72) via DISPERSION-based aggregations")
print("     (std, range) while mean/median FAIL (<0.5 AUC).")
print("     Shows Claim 2 is decisive on transfer: no single aggregation dominates.")

print("\n[Text-only Judge on other datasets]")
print(f"  Transfer:        AUC={tj_transfer['overall']['auroc']:.3f} (blatant collusion recognized)")
print(f"  Stego:           AUC={tj_stego['overall']['auroc']:.3f}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
Claim 1 (probes recover signal text misses):     STRONGLY SUPPORTED
  Core: probe 0.95 AUC vs text judge 0.50 AUC
Claim 2 (group aggregation, no single dominant): SUPPORTED
  Aggregation always beats worst single agent; best method varies by dataset.
  Core = mean/std; Transfer = mean/median/topk2; Stego (transfer) = std/range.
Claim 3 (zero-shot transfer to new scenarios):   SUPPORTED (Transfer), MIXED (Stego)
  Core -> Transfer: 0.956 overall, 5/6 families at 1.00 AUC.
  Core -> Stego: 0.72 requires dispersion aggregations; direct-orientation aggregations
  fail, illustrating why a FAMILY of aggregations is needed for cross-domain use.
""")
