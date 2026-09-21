"""ESMFold trunk interventions: capture per-block s,z; patch s or z at block k.

The trunk is num_blocks=48 iterated blocks that update (s, z) jointly.
Interventions:
  - s_patch(k, target_positions): after block k finishes, overwrite s at
    target_positions with source's s[k, target_positions].
  - z_patch(k, pair_positions): after block k finishes, overwrite z at
    pair_positions x pair_positions with source's z[k, pair, pair].
  - seq2pair_ablate(k, seq_positions): monkey-patch block k's
    sequence_to_pair.forward to zero-out contributions at seq_positions.
  - pair2seq_ablate(k, seq_positions): monkey-patch block k's
    pair_to_sequence.forward to zero-out attention bias at seq_positions.
"""
from contextlib import contextmanager
from typing import Optional, Sequence
import torch


@contextmanager
def capture_block_outputs(model, save):
    """During forward pass, record (s, z) after each block."""
    hooks = []
    for i, blk in enumerate(model.trunk.blocks):
        def make(i):
            def hook(mod, inp, out):
                s, z = out
                save[i] = (s.detach().clone(), z.detach().clone())
            return hook
        hooks.append(blk.register_forward_hook(make(i)))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def patch_s_at_block(model, k, s_source, positions=None):
    """After block k finishes, overwrite s at `positions` with `s_source` (same shape as full s or slice for positions)."""
    def hook(mod, inp, out):
        s, z = out
        s = s.clone()
        if positions is None:
            s = s_source.to(s.device, s.dtype).clone()
        else:
            s[:, positions, :] = s_source.to(s.device, s.dtype)
        return (s, z)
    h = model.trunk.blocks[k].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def patch_z_at_block(model, k, z_source, pair_positions=None):
    """After block k, overwrite z at pair_positions×pair_positions with z_source."""
    def hook(mod, inp, out):
        s, z = out
        z = z.clone()
        if pair_positions is None:
            z = z_source.to(z.device, z.dtype).clone()
        else:
            pos = torch.tensor(pair_positions, device=z.device, dtype=torch.long)
            # z_source is expected to be (B, |pos|, |pos|, C) — use ix_ style
            idx_i = pos[:, None].expand(-1, len(pair_positions))
            idx_j = pos[None, :].expand(len(pair_positions), -1)
            z[:, idx_i, idx_j, :] = z_source.to(z.device, z.dtype)
        return (s, z)
    h = model.trunk.blocks[k].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def patch_z_full_at_block(model, k, z_source):
    """After block k, replace full pairwise state with z_source."""
    def hook(mod, inp, out):
        s, z = out
        return (s, z_source.to(z.device, z.dtype).clone())
    h = model.trunk.blocks[k].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def ablate_seq2pair(model, block_range, mode="zero"):
    """Zero out the sequence_to_pair contribution in specific blocks.

    Replace `block.sequence_to_pair(sequence_state)` with zeros of same shape.
    """
    handles = []
    for k in block_range:
        s2p = model.trunk.blocks[k].sequence_to_pair
        orig_forward = s2p.forward

        def make(orig, s2p_mod):
            def new_forward(sequence_state, *a, **kw):
                out = orig(sequence_state, *a, **kw)
                if mode == "zero":
                    return torch.zeros_like(out)
                raise ValueError(mode)
            return new_forward
        s2p.forward = make(orig_forward, s2p)
        handles.append((s2p, orig_forward))
    try:
        yield
    finally:
        for mod, orig in handles:
            mod.forward = orig


@contextmanager
def ablate_pair2seq(model, block_range, mode="zero"):
    """Zero out the pair_to_sequence contribution (attention bias) in specific blocks."""
    handles = []
    for k in block_range:
        p2s = model.trunk.blocks[k].pair_to_sequence
        orig_forward = p2s.forward

        def make(orig, p2s_mod):
            def new_forward(pairwise_state, *a, **kw):
                out = orig(pairwise_state, *a, **kw)
                if mode == "zero":
                    return torch.zeros_like(out)
                raise ValueError(mode)
            return new_forward
        p2s.forward = make(orig_forward, p2s)
        handles.append((p2s, orig_forward))
    try:
        yield
    finally:
        for mod, orig in handles:
            mod.forward = orig


@contextmanager
def add_direction_to_s(model, block_range, direction_vec, positions, scale=1.0):
    """After each specified block, add `scale * direction_vec` to s at `positions`.
    direction_vec: (D,) or (|positions|, D) tensor.
    """
    handles = []
    for k in block_range:
        blk = model.trunk.blocks[k]
        def make(k):
            def hook(mod, inp, out):
                s, z = out
                s = s.clone()
                pos = torch.tensor(positions, device=s.device, dtype=torch.long)
                v = direction_vec.to(s.device, s.dtype)
                if v.dim() == 1:
                    s[:, pos, :] = s[:, pos, :] + scale * v
                else:
                    s[:, pos, :] = s[:, pos, :] + scale * v
                return (s, z)
            return hook
        handles.append(blk.register_forward_hook(make(k)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
