"""
Steering methods for the protein-structure figure pipeline (isolated). Two intervention families,
both exposing the SAME interface as harness2.CSteerer (`.steer(dose)` context manager +
`.last_impact`), so they drop straight into harness2.generate_proteins / fold_records /
aggregate_predictor.

  AmplifySteerer      -- open-loop feature amplification (== the main-experiment mechanism,
                         raw-alpha dosing):
                         delta = dose * base_dir,   base_dir = sum_{f in S} s_f * W[:,f].
                         (Kept here only for parity; the main experiment itself runs on the
                         frozen harness2.CSteerer path.)

  ClampDecodeSteerer  -- closed-loop Encoder-Clamp-Decoder:
                         z = topk_relu( SAE.encode(x) )            # READ current feature acts
                         z_S <- dose * s_f                          # CLAMP target features (pin)
                         mode="delta"  (primary): xnew = x + (z_clamp - z)_S @ W[:,S]^T
                             error-preserving -- injects ONLY the change from clamping, keeps the
                             rest of the residual (incl. SAE reconstruction error) intact.
                         mode="replace": xnew = SAE.decode(clamp(SAE.encode(x)))
                             literal full reconstruction replacement (dumps recon error into stream).

Dose semantics (both methods share the raw-alpha-style axis): clamp target for feature f is
`dose * s_f` where s_f is f's mean nonzero (natural) activation. dose=0 -> hook no-ops (true
baseline), so the c=0 point is a within-machinery control identical to the main experiment.
"""
import os, sys, math, contextlib
import numpy as np
import torch

sys.path.insert(0, "/data/wanghaoxiong/intergene_mechanist_v6/code")
from evo2_sae import BatchTopKSAE, HIDDEN
from mechanism import STEER_SITE

SQRT_D = math.sqrt(HIDDEN)


class AmplifySteerer:
    """Open-loop amplification (raw-alpha). delta = dose * base_dir. Interface-compatible."""

    def __init__(self, model, sae: BatchTopKSAE, feats, s_f, device="cuda:0"):
        self.model, self.sae, self.device = model, sae, device
        self.dose = 0.0
        self._handle = None
        self.last_impact = None
        self.set_direction(feats, s_f)

    def set_direction(self, feats, s_f):
        self.feats = torch.as_tensor(list(feats), dtype=torch.long, device=self.device)
        self.s_f = torch.as_tensor(list(s_f), dtype=torch.float32, device=self.device)
        Wc = self.sae.W[:, self.feats].float()
        self.base_dir = (Wc * self.s_f.unsqueeze(0)).sum(1)   # (4096,)
        self.base_norm = float(self.base_dir.norm().item())

    def set_random_norm_matched(self, feats, s_f, target_base_norm):
        self.set_direction(feats, s_f)
        self.base_dir = self.base_dir * (target_base_norm / (self.base_norm + 1e-8))
        self.base_norm = float(self.base_dir.norm().item())

    def _hook(self, module, inputs, output):
        if self.dose == 0.0:
            return output
        x = output[0] if isinstance(output, tuple) else output
        xf = x.float()
        delta = self.dose * self.base_dir
        xn = xf.norm(dim=-1)
        self.last_impact = float((delta.norm() / (xn.mean() + 1e-8)).item())
        xnew = (xf + delta).to(x.dtype)
        return (xnew,) + tuple(output[1:]) if isinstance(output, tuple) else xnew

    @contextlib.contextmanager
    def steer(self, dose):
        self.dose = float(dose)
        self.last_impact = None
        blk = self.model.model.get_submodule(STEER_SITE)
        self._handle = blk.register_forward_hook(self._hook)
        try:
            yield
        finally:
            if self._handle:
                self._handle.remove(); self._handle = None
            self.dose = 0.0


class ClampDecodeSteerer:
    """Encoder-Clamp-Decoder steering at blocks.26.post_norm."""

    def __init__(self, model, sae: BatchTopKSAE, feats, s_f, device="cuda:0", mode="delta"):
        assert mode in ("delta", "replace")
        self.model, self.sae, self.device, self.mode = model, sae, device, mode
        self.dose = 0.0
        self._handle = None
        self.last_impact = None
        self.set_direction(feats, s_f)

    def set_direction(self, feats, s_f):
        self.feats = torch.as_tensor(list(feats), dtype=torch.long, device=self.device)
        self.s_f = torch.as_tensor(list(s_f), dtype=torch.float32, device=self.device)   # (|S|,)
        self.Wc = self.sae.W[:, self.feats].float()                                       # (4096,|S|)
        # leading-order steering direction (== amplification base_dir), for norm-matching the null
        self.base_dir = (self.Wc * self.s_f.unsqueeze(0)).sum(1)
        self.base_norm = float(self.base_dir.norm().item())

    def set_random_norm_matched(self, feats, s_f, target_base_norm):
        """Null direction: |S| random features, rescale s_f so the leading-order steering norm
        (||sum_f s_f W[:,f]||) matches the target set -> dose axis shared at leading order."""
        feats = list(feats); s_f = list(s_f)
        Wc = self.sae.W[:, torch.as_tensor(feats, dtype=torch.long, device=self.device)].float()
        sf = torch.as_tensor(s_f, dtype=torch.float32, device=self.device)
        bn = float((Wc * sf.unsqueeze(0)).sum(1).norm().item())
        scale = target_base_norm / (bn + 1e-8)
        self.set_direction(feats, [v * scale for v in s_f])

    def _hook(self, module, inputs, output):
        if self.dose == 0.0:
            return output
        x = output[0] if isinstance(output, tuple) else output
        xf = x.float()
        z = self.sae.encode(xf)                       # (B,L,32768) topk-relu current feature acts
        z_cur = z[..., self.feats]                    # (B,L,|S|)
        target = self.dose * self.s_f                 # (|S|,) pinned clamp value
        if self.mode == "delta":
            dz = target.unsqueeze(0).unsqueeze(0) - z_cur          # (B,L,|S|)
            delta = dz @ self.Wc.t()                               # (B,L,4096) error-preserving
            xnew = xf + delta
        else:  # replace: full SAE reconstruction with S pinned
            z2 = z.clone()
            z2[..., self.feats] = target
            xnew = self.sae.decode(z2)
            delta = xnew - xf
        xn = xf.norm(dim=-1)
        self.last_impact = float((delta.norm(dim=-1).mean() / (xn.mean() + 1e-8)).item())
        xnew = xnew.to(x.dtype)
        return (xnew,) + tuple(output[1:]) if isinstance(output, tuple) else xnew

    @contextlib.contextmanager
    def steer(self, dose):
        self.dose = float(dose)
        self.last_impact = None
        blk = self.model.model.get_submodule(STEER_SITE)
        self._handle = blk.register_forward_hook(self._hook)
        try:
            yield
        finally:
            if self._handle:
                self._handle.remove(); self._handle = None
            self.dose = 0.0


def make_steerer(method, model, sae, feats, s_f, device="cuda:0", clamp_mode="delta"):
    if method == "amplify":
        return AmplifySteerer(model, sae, feats, s_f, device=device)
    if method == "clampdecode":
        return ClampDecodeSteerer(model, sae, feats, s_f, device=device, mode=clamp_mode)
    raise ValueError(method)
