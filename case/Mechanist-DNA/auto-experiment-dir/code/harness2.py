"""
Round-2 steering + readout harness (builds on round-1 mechanism.py, hardened).

Round-2 changes vs round-1:
  * sigma_proj-unit dosing (plan improvement 4): steered activation
        x' = x + (c * sigma_proj) * unit_dir(S),   unit_dir(S) = base_dir / ||base_dir||
    where base_dir = sum_{f in S} s_f * W[:,f] (SAE tied decoder columns), and sigma_proj is the
    std of the natural-CDS activation projected onto unit_dir(S) at blocks.26.post_norm. The raw-a
    round-1 hook did  x' = x + a * base_dir, so  a <-> c  via  a = c * sigma_proj / ||base_dir||.
  * capability/impact metrics recorded per generation (plan P4): valid-ORF + mean steering
    perturbation ratio ||delta|| / ||x|| at the site (a cheap logit/representation-drift proxy).
  * generation is FROZEN without any structure predictor; both predictors fold the SAME sequences
    afterwards (plan P3/P8). See fold.fold_and_read for the dual-predictor readout.

Everything a structure predictor needs (translation, ORF filter) is shared with round-1
mechanism.py; this module only adds the sigma_proj dosing + capability logging + dual-predictor glue.
"""
import os, sys, math, json, contextlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from evo2_sae import BatchTopKSAE, HIDDEN, SAE_DICT
import mechanism as M            # translate_orf, generate_dna, STEER_SITE
from mechanism import STEER_SITE, translate_orf, generate_dna

SQRT_D = math.sqrt(HIDDEN)


# ======================================================================================
# sigma_proj-unit steerer
# ======================================================================================
class CSteerer:
    """Amplify a feature set at blocks.26.post_norm in sigma_proj units.

    dose c is in sigma_proj units. Internally applies delta = alpha_raw * base_dir with
        alpha_raw = c * sigma_proj / ||base_dir||     (so ||delta|| = c * sigma_proj)
    which is identical to round-1's raw-a hook at a = alpha_raw. base_dir / sigma_proj can be
    overridden (random-direction null) via set_direction().
    """

    def __init__(self, model, sae: BatchTopKSAE, feats, s_f, sigma_proj=None, device="cuda:0"):
        self.model = model
        self.sae = sae
        self.device = device
        self.set_direction(feats, s_f, sigma_proj)
        self.c = 0.0
        self._handle = None
        self.last_impact = None   # (mean ||delta||/||x||) captured during the steered forward

    def set_direction(self, feats, s_f, sigma_proj=None):
        self.feats = torch.as_tensor(list(feats), dtype=torch.long, device=self.device)
        self.s_f = torch.as_tensor(list(s_f), dtype=torch.float32, device=self.device)
        Wc = self.sae.W[:, self.feats].float()          # (4096, |S|)
        self.base_dir = (Wc * self.s_f.unsqueeze(0)).sum(1)   # (4096,)
        self.base_norm = float(self.base_dir.norm().item())
        self.unit_dir = self.base_dir / (self.base_norm + 1e-8)
        self.sigma_proj = float(sigma_proj) if sigma_proj is not None else self.base_norm
        # alpha_raw multiplier s.t. delta = (c*sigma_proj) * unit_dir = alpha_raw * base_dir
        self._alpha_per_c = self.sigma_proj / (self.base_norm + 1e-8)

    def set_norm_matched_direction(self, feats, s_f, target_base_norm, sigma_proj):
        """Random-direction null: build base_dir from feats, rescale to match S's ||base_dir||,
        keep S's sigma_proj so the dose axis is shared (norm-matched, sigma_proj-consistent)."""
        self.feats = torch.as_tensor(list(feats), dtype=torch.long, device=self.device)
        self.s_f = torch.as_tensor(list(s_f), dtype=torch.float32, device=self.device)
        Wc = self.sae.W[:, self.feats].float()
        base = (Wc * self.s_f.unsqueeze(0)).sum(1)
        base = base * (target_base_norm / (float(base.norm().item()) + 1e-8))
        self.base_dir = base
        self.base_norm = float(base.norm().item())
        self.unit_dir = self.base_dir / (self.base_norm + 1e-8)
        self.sigma_proj = float(sigma_proj)
        self._alpha_per_c = self.sigma_proj / (self.base_norm + 1e-8)

    def _hook(self, module, inputs, output):
        if self.c == 0.0:
            return output
        x = output[0] if isinstance(output, tuple) else output
        xf = x.float()
        alpha_raw = self.c * self._alpha_per_c
        delta = alpha_raw * self.base_dir                   # (4096,)
        # capability/impact proxy: mean ||delta|| / ||x|| over tokens
        xn = xf.norm(dim=-1)
        self.last_impact = float((delta.norm() / (xn.mean() + 1e-8)).item())
        xnew = (xf + delta).to(x.dtype)
        if isinstance(output, tuple):
            return (xnew,) + tuple(output[1:])
        return xnew

    @contextlib.contextmanager
    def steer(self, c):
        self.c = float(c)
        self.last_impact = None
        blk = self.model.model.get_submodule(STEER_SITE)
        self._handle = blk.register_forward_hook(self._hook)
        try:
            yield
        finally:
            if self._handle:
                self._handle.remove()
                self._handle = None
            self.c = 0.0


# ======================================================================================
# sigma_proj calibration
# ======================================================================================
@torch.no_grad()
def compute_sigma_proj(model, sae, feats, s_f, ref_seqs, device="cuda:0"):
    """sigma_proj = std over natural-CDS reference codons of the activation projected onto
    unit_dir(S) at blocks.26.post_norm. Also returns ||base_dir||."""
    Wc = sae.W[:, torch.as_tensor(feats, dtype=torch.long, device=device)].float()
    sf = torch.as_tensor(s_f, dtype=torch.float32, device=device)
    base_dir = (Wc * sf.unsqueeze(0)).sum(1)
    base_norm = float(base_dir.norm().item())
    unit = base_dir / (base_norm + 1e-8)
    projs = []
    site = model.model.get_submodule(STEER_SITE)
    cap = {}

    def hook(mod, inp, out):
        cap["x"] = (out[0] if isinstance(out, tuple) else out).detach().float()

    h = site.register_forward_hook(hook)
    try:
        for seq in ref_seqs:
            ids = torch.tensor([model.tokenizer.tokenize(seq)], dtype=torch.long, device=device)
            model(ids)
            x = cap["x"][0]                       # (L, 4096)
            projs.append((x @ unit).cpu().numpy())
    finally:
        h.remove()
    allp = np.concatenate(projs)
    return {"sigma_proj": float(allp.std()), "base_norm": base_norm,
            "proj_mean": float(allp.mean()), "n_ref_codons": int(allp.size)}


# ======================================================================================
# generation (Evo2 only; predictors run afterwards) + dual-predictor readout
# ======================================================================================
@torch.no_grad()
def generate_proteins(evo2, steerer, prompts, c, n_tokens, temperature, top_k, seed_base,
                      table, min_aa=30):
    """Steered Evo2 generation of one sequence per prompt at dose c (sigma_proj units).
    Returns list of per-sample dicts: {idx, valid_orf, len_aa, prot(str|None), impact}."""
    records = []
    # Always register the hook (it no-ops at c==0), so the c=0 baseline shares the exact
    # intervention code path -- zero-dose is a true within-machinery control, not a separate path.
    impacts = []
    with steerer.steer(c):
        for i, prompt in enumerate(prompts):
            rec = {"idx": i, "valid_orf": False, "prot": None, "len_aa": 0}
            try:
                full = generate_dna(evo2, prompt, n_tokens, temperature, top_k, seed=seed_base + i)
                gen = full[len(prompt):] if full.startswith(prompt) else full
                prot, valid, meta = translate_orf(prompt + gen, min_aa=min_aa, table=table)
                rec.update({"valid_orf": valid, "len_aa": meta.get("len_aa", 0),
                            "had_internal_stop": meta.get("had_internal_stop", False),
                            "prot": prot if valid else None})
            except Exception as e:
                rec["error"] = f"gen: {type(e).__name__}: {str(e)[:80]}"
            if steerer.last_impact is not None:
                impacts.append(steerer.last_impact)
            records.append(rec)
    impact_mean = float(np.mean(impacts)) if impacts else 0.0
    for r in records:
        r["impact_ratio"] = impact_mean
    return records


def fold_records(records, predictor, device="cuda:0", num_cycle=None, max_len=400):
    """Fold each valid-ORF protein with `predictor` and attach the structural readout in-place
    under key `struct_<predictor>`. Returns nothing (mutates records)."""
    from fold import fold_and_read
    for rec in records:
        if not rec.get("prot"):
            continue
        rd = fold_and_read(rec["prot"], predictor=predictor, device=device,
                           num_cycle=num_cycle, max_len=max_len)
        rec[f"struct_{predictor}"] = rd   # dict or None


# ======================================================================================
# aggregation with pLDDT-weighted (PRIMARY) + hard-gated + threshold-sweep endpoints
# ======================================================================================
PLDDT_GRID = (50, 60, 70, 80)


def aggregate_predictor(records, predictor, key="helix_hgi_w"):
    """Aggregate one predictor's readout across a run's records.
    Primary key defaults to the pLDDT-WEIGHTED helix fraction ('helix_hgi_w')."""
    sk = f"struct_{predictor}"
    folded = [r[sk] for r in records if r.get(sk)]
    n = len(records)
    n_valid = sum(r.get("valid_orf", False) for r in records)
    out = {
        "predictor": predictor, "n": n, "n_valid_orf": n_valid,
        "valid_orf_rate": n_valid / max(n, 1),
        "n_folded": len(folded), "folded_rate": len(folded) / max(n, 1),
        "impact_ratio": float(np.mean([r.get("impact_ratio", 0.0) for r in records])) if n else 0.0,
    }
    if not folded:
        return out
    def col(k):
        return np.array([f[k] for f in folded if f.get(k) is not None], dtype=float)
    for k in ("helix_hgi_w", "helix_hgi", "helix_h_w", "helix_h", "sheet_w", "sheet",
              "mean_plddt", "struct_mean_plddt"):
        v = col(k)
        if v.size:
            out[f"{k}_mean"] = float(v.mean())
            out[f"{k}_std"] = float(v.std())
            out[f"{k}_sem"] = float(v.std() / math.sqrt(v.size))
    for T in PLDDT_GRID:
        v = col(f"helix_hgi_gate{T}")
        if v.size:
            out[f"helix_hgi_gate{T}_mean"] = float(v.mean())
        v2 = col(f"sheet_gate{T}")
        if v2.size:
            out[f"sheet_gate{T}_mean"] = float(v2.mean())
    # keep the per-sample PRIMARY endpoint + prompt idx for downstream cluster-bootstrap / trend
    out["per_sample"] = [
        {"idx": r["idx"], "prompt_cluster": r.get("prompt_cluster", r["idx"]),
         key: r[sk].get(key), "helix_hgi": r[sk].get("helix_hgi"),
         "sheet_w": r[sk].get("sheet_w"), "sheet": r[sk].get("sheet"),
         "plddt": r[sk].get("struct_mean_plddt"), "valid_orf": r.get("valid_orf")}
        for r in records if r.get(sk)
    ]
    return out


# ======================================================================================
# stats helpers (downstream, plan P2/P5)
# ======================================================================================
def cluster_bootstrap_ci(values, clusters, n_boot=2000, seed=0, alpha=0.05):
    """Cluster-robust bootstrap CI of the mean: resample CLUSTERS (prompt-seed families), not
    individual ORFs (plan P5 effective-N). Returns (mean, lo, hi, n_clusters)."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if values.size == 0:
        return (None, None, None, 0)
    uniq = np.unique(clusters)
    by = {c: values[clusters == c] for c in uniq}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        boot[b] = np.concatenate([by[c] for c in pick]).mean()
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(values.mean()), float(lo), float(hi), int(len(uniq)))


def cluster_bootstrap_diff(vals_a, clus_a, vals_b, clus_b, n_boot=2000, seed=0, alpha=0.05):
    """Cluster-robust bootstrap CI of a mean DIFFERENCE (arm A - arm B), resampling the SHARED
    prompt clusters JOINTLY (plan P5). A and B are matched on prompt cluster (same prompt set
    across arms, P8), so the contrast is per-cluster (mean_A[c] - mean_B[c]) then bootstrapped
    over clusters. Returns (diff_mean, lo, hi, n_clusters)."""
    va, ca = np.asarray(vals_a, float), np.asarray(clus_a)
    vb, cb = np.asarray(vals_b, float), np.asarray(clus_b)
    if va.size == 0 or vb.size == 0:
        return (None, None, None, 0)
    ma = {c: va[ca == c].mean() for c in np.unique(ca)}
    mb = {c: vb[cb == c].mean() for c in np.unique(cb)}
    shared = sorted(set(ma) & set(mb))
    if not shared:
        return (None, None, None, 0)
    d = np.array([ma[c] - mb[c] for c in shared])
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    idx = np.arange(len(shared))
    for b in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        boot[b] = d[pick].mean()
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(d.mean()), float(lo), float(hi), int(len(shared)))


def spearman_trend(x, y):
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return float(r), float(p)


def bh_fdr(pvals, q=0.05):
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        adj[i] = prev
    return (adj <= q), adj
