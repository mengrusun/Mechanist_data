"""List available pythia-1b intermediate checkpoints."""
import os
root = "/mnt/quarkfs/share_model/Ptyhia/pythia-1b-checkpoints"
steps = sorted([int(d[4:]) for d in os.listdir(root) if d.startswith("step")])
print(",".join(str(s) for s in steps))
print("count =", len(steps))
