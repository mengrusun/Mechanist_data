"""Print a compact per-layer table and per-behaviour best-layer summary."""
import json, sys
with open("results/probe_accuracy.json") as f: probe = json.load(f)

behaviours = list(probe.keys())
L_max = max(int(L) for L in probe[behaviours[0]])

print(f"{'layer':>5} | " + " | ".join(f"{b:>25}" for b in behaviours))
print("-" * (7 + 30 * len(behaviours)))
for L in range(L_max + 1):
    row = [f"{L:>5}"]
    for b in behaviours:
        e = probe[b][str(L)]
        row.append(f"acc={e['val_acc']:.2f} gap={e['sep_gap']:>5.2f} |n|={e['dir_norm']:>5.2f}".rjust(25))
    print(row[0] + " | " + " | ".join(row[1:]))

print("\n=== Number of layers with val_acc >= 0.95 ===")
for b in behaviours:
    n95 = sum(1 for L in probe[b] if probe[b][L]["val_acc"] >= 0.95)
    n100 = sum(1 for L in probe[b] if probe[b][L]["val_acc"] >= 0.999)
    print(f"  {b}: {n95} layers >=0.95   {n100} layers ==1.00   out of {L_max+1}")
