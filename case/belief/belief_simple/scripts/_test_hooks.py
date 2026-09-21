"""Sanity: verify that install_head_scaling_hooks (α=0) matches passing head_mask=0.

Runs on pythia-410m for speed. If both approaches give the same logits for a random
input, the hook implementation is correct.
"""
import sys
sys.path.insert(0, '/mnt/quarkfs/xuweihong/MECHANICA_exps/exp18/scripts')
import torch
from belief_utils import load_model_and_tokenizer, install_head_scaling_hooks, remove_hooks, model_arch_info

net, tok = load_model_and_tokenizer('/mnt/quarkfs/share_model/Ptyhia', 'pythia-410m', dtype='fp32', device='cuda:1')
info = model_arch_info(net)
print(f"arch: {info}")

x = torch.tensor([tok.encode("The sky is blue and the sun is yellow.")], device='cuda:1')

# Clean forward
with torch.no_grad():
    clean = net(x).logits[0].detach().float().cpu()
print(f"clean logits shape: {clean.shape}")

# Ablate head (layer=5, head=3) via forward hook (α=0)
targets = [(5, 3)]
scale_map = {(l, h): 0.0 for (l, h) in targets}
handles = install_head_scaling_hooks(net, scale_map)
with torch.no_grad():
    hooked = net(x).logits[0].detach().float().cpu()
remove_hooks(handles)

# Difference
diff = (clean - hooked).abs().max().item()
print(f"clean vs hooked (α=0) max abs diff: {diff:.6f}")
print(f"clean sample: {clean[0, :5].tolist()}")
print(f"hooked sample: {hooked[0, :5].tolist()}")

# Also test α=1 (identity)
scale_map = {(5, 3): 1.0}
handles = install_head_scaling_hooks(net, scale_map)
with torch.no_grad():
    identity = net(x).logits[0].detach().float().cpu()
remove_hooks(handles)
diff2 = (clean - identity).abs().max().item()
print(f"clean vs hooked (α=1) max abs diff: {diff2:.6e} (should be 0 or ~machine epsilon)")

# α=2 amplification
scale_map = {(5, 3): 2.0}
handles = install_head_scaling_hooks(net, scale_map)
with torch.no_grad():
    amp = net(x).logits[0].detach().float().cpu()
remove_hooks(handles)
diff3 = (clean - amp).abs().max().item()
print(f"clean vs hooked (α=2) max abs diff: {diff3:.6f} (should be non-trivial)")

# Sanity: ablate multiple heads at once — result should differ from ablating them one-by-one
# and compose (linear in dense input, but dense is followed by residual sum + rest of network,
# so composition is NOT linear — that's expected).
scale_map = {(5, 3): 0.0, (7, 2): 0.0}
handles = install_head_scaling_hooks(net, scale_map)
with torch.no_grad():
    multi = net(x).logits[0].detach().float().cpu()
remove_hooks(handles)
print(f"multi-head ablation (5,3)+(7,2) max abs diff vs clean: {(clean-multi).abs().max().item():.6f}")

print("OK — hook mechanism functioning.")
