"""Core generation + steering + readout for M2 / M-CTRL / M3.

Given the frozen feature set S (results/m1_features.json), generate DNA under a
steering coefficient alpha (residual-add of Sum_i s_i * dhat_i at blocks.26),
extract the longest ORF, treatment-independent QC, Evo2 perplexity, then fold the
translated ORF with ESMFold and DSSP -> predicted %helix.

Reused by:
  m2_generate_eval.py  (fixed n, dev seed block D, alpha grid + control arms)
  m3_confirm.py        (target #valid, held-out seed block H)
"""
import os, sys, json, time, argparse, hashlib
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import common as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ITT value assigned to QC-failed sequences (prespecified): helix = 0.0
ITT_FAIL_HELIX = 0.0


# ------------------------------------------------------------------ steer vectors
def load_features(path):
    with open(path) as fh:
        return json.load(fh)

def build_steer_vector(feat, sae, dev, control=None, seed=0, k_for_control=None):
    """Return (vec (4096,) on dev, meta dict). control in
    {None,'random_feature','beta_sheet','null_direction'}."""
    S = np.array(feat["S"], dtype=np.int64)
    s_i = np.array([feat["s_i"][str(i)] for i in S], dtype=np.float32)
    v_S = sae.steer_vector(torch.tensor(S, device=sae.device),
                           torch.tensor(s_i, device=sae.device))
    norm_S = float(v_S.norm().item())
    rng = np.random.default_rng(seed)
    K = len(S) if k_for_control is None else k_for_control

    if control is None:
        return v_S.to(dev), dict(kind="S", latents=S.tolist(), norm=norm_S)

    if control == "beta_sheet":
        beta = np.array(feat.get("beta_top", []), dtype=np.int64)[:K]
        # s_i for beta latents: median positive train activation is unavailable here;
        # reuse decoder-direction with unit s -> then rescale to match norm_S.
        v = sae.steer_vector(torch.tensor(beta, device=sae.device),
                             torch.ones(len(beta), device=sae.device))
        v = v * (norm_S / (float(v.norm().item()) + 1e-8))
        return v.to(dev), dict(kind="beta_sheet", latents=beta.tolist(), norm=norm_S)

    if control == "random_feature":
        # K random latents (excluding S), matched to norm_S
        pool = np.setdiff1d(np.arange(C.D_SAE), S)
        rlat = rng.choice(pool, size=K, replace=False)
        v = sae.steer_vector(torch.tensor(rlat, device=sae.device),
                             torch.ones(len(rlat), device=sae.device))
        v = v * (norm_S / (float(v.norm().item()) + 1e-8))
        return v.to(dev), dict(kind="random_feature", latents=rlat.tolist(), norm=norm_S)

    if control == "null_direction":
        r = torch.tensor(rng.standard_normal(C.D_MODEL), device=sae.device, dtype=sae.dtype)
        r = r / r.norm() * norm_S
        return r.to(dev), dict(kind="null_direction", norm=norm_S)

    raise ValueError(control)


# ------------------------------------------------------------------ perplexity
@torch.no_grad()
def evo2_perplexity(evo2, dna_list, batch_size=8):
    """Mean-per-nucleotide perplexity under the (unsteered) model."""
    if not dna_list:
        return []
    scores = evo2.score_sequences(dna_list, batch_size=batch_size,
                                  reduce_method="mean")   # mean logprob/nt
    return [float(np.exp(-s)) for s in scores]


# ------------------------------------------------------------------ generation
@torch.no_grad()
def generate_arm(evo2, vec, alpha, seeds, dev, n_tokens=C.GEN_MAX_NT, batch=32):
    """Generate one sequence per seed (paired). Returns list of raw DNA strings.
    Deterministic per seed via torch manual seed before each batch."""
    seqs = []
    for b0 in range(0, len(seeds), batch):
        bseeds = seeds[b0:b0+batch]
        # seed the RNG deterministically from the batch's seed list (bounded < 2^32)
        C.set_seed((int(bseeds[0]) * 100003 + len(bseeds)) % (2**32 - 1))
        prompts = [C.START_CONTEXT] * len(bseeds)
        with C.ResidualSteerer(evo2, vec, alpha=alpha):
            out = evo2.generate(prompts, n_tokens=n_tokens,
                                temperature=C.DEC_TEMPERATURE, top_k=C.DEC_TOP_K,
                                top_p=C.DEC_TOP_P, cached_generation=True, verbose=0)
        for s in out.sequences:
            seqs.append(C.START_CONTEXT + s)
    return seqs


# ------------------------------------------------------------------ one arm eval
def eval_arm(evo2, folder, vec, alpha, seeds, dev, meta, telemetry_vec=None,
             gen_batch=32, fold_max=None, log=print):
    """Generate + QC + perplexity + fold + DSSP for one arm. Returns record dict."""
    t0 = time.time()
    raw = generate_arm(evo2, vec, alpha, seeds, dev, batch=gen_batch)
    per = []
    valid_dna, valid_ix = [], []
    for i, dna in enumerate(raw):
        prot, olen, frame, st = C.longest_orf_protein(dna)
        ok, why = C.qc_valid(prot, olen) if prot else (False, "no_orf")
        n_bad = sum(1 for ch in dna.upper() if ch not in "ACGT")
        rec = dict(seed=int(seeds[i]), orf_len_nt=int(olen), prot_len=len(prot),
                   qc_ok=bool(ok), qc_reason=why, gc=C.gc_content(dna),
                   len_raw=len(dna), non_acgt_frac=n_bad / max(1, len(dna)),
                   protein=prot if ok else "")
        per.append(rec)
        if ok:
            valid_dna.append(dna); valid_ix.append(i)
    log(f"[arm a={alpha} {meta.get('kind')}] generated {len(raw)} "
        f"valid_orf={len(valid_ix)} ({time.time()-t0:.0f}s)")

    # perplexity on the full generated DNA (fluency / validity floor)
    ppls = evo2_perplexity(evo2, [raw[i] for i in range(len(raw))], batch_size=8)
    for i, p in enumerate(ppls):
        per[i]["ppl"] = p

    # fold valid ORFs
    fold_ix = valid_ix if fold_max is None else valid_ix[:fold_max]
    tf = time.time()
    for j, i in enumerate(fold_ix):
        try:
            r = folder.helix_readout(per[i]["protein"])
        except Exception as e:
            per[i]["fold_err"] = str(e)[:120]
            continue
        per[i].update({f"ss_{k}": v for k, v in r.items() if k != "ss3"})
        per[i]["helix_all"] = r["helix_all"]
        per[i]["helix_hi"] = r["helix_hi"]
        per[i]["sheet_all"] = r["sheet_all"]
        per[i]["coil_all"] = r["coil_all"]
        per[i]["mean_plddt"] = r["mean_plddt"]
        per[i]["n_hi"] = r["n_hi"]
        if (j+1) % 50 == 0:
            log(f"[arm a={alpha}] folded {j+1}/{len(fold_ix)} ({time.time()-tf:.0f}s)")
    log(f"[arm a={alpha} {meta.get('kind')}] folded {len(fold_ix)} in {time.time()-tf:.0f}s")

    # steering telemetry: realized target-latent activation change is captured
    # separately (optional); record the steer-vector norm and alpha here.
    arm = dict(alpha=float(alpha), meta=meta, n_gen=len(raw),
               n_valid=len(valid_ix), per_seq=per,
               steer_norm=float(meta.get("norm", 0.0)),
               wall_s=time.time()-t0)
    return arm


# ------------------------------------------------------------------ arm stats
def arm_summary(arm):
    per = arm["per_seq"]
    valid = [p for p in per if p.get("qc_ok") and ("helix_all" in p) and
             p["helix_all"] == p["helix_all"]]  # not nan
    hi_valid = [p for p in valid if p.get("n_hi", 0) >= 20 and
                p.get("helix_hi") == p.get("helix_hi")]
    def m(xs): return float(np.mean(xs)) if xs else float("nan")
    # ITT: all generated; failures -> ITT_FAIL_HELIX
    itt_vals = []
    for p in per:
        if p.get("qc_ok") and ("helix_all" in p) and p["helix_all"] == p["helix_all"]:
            itt_vals.append(p["helix_all"])
        else:
            itt_vals.append(ITT_FAIL_HELIX)
    cond = [p["helix_all"] for p in valid]
    ppl_all = [p["ppl"] for p in per if "ppl" in p and p["ppl"] == p["ppl"]]
    return dict(
        n_gen=arm["n_gen"], n_valid=len(valid),
        valid_rate=len(valid) / max(1, arm["n_gen"]),
        helix_cond_mean=m(cond), helix_itt_mean=m(itt_vals),
        helix_hi_cond_mean=m([p["helix_hi"] for p in hi_valid]),
        sheet_cond_mean=m([p.get("sheet_all") for p in valid
                           if p.get("sheet_all") == p.get("sheet_all")]),
        coil_cond_mean=m([p.get("coil_all") for p in valid
                          if p.get("coil_all") == p.get("coil_all")]),
        plddt_mean=m([p.get("mean_plddt") for p in valid
                      if p.get("mean_plddt") == p.get("mean_plddt")]),
        ppl_median=float(np.median(ppl_all)) if ppl_all else float("nan"),
        ppl_p95=float(np.percentile(ppl_all, 95)) if ppl_all else float("nan"),
        gc_mean=m([p["gc"] for p in per]),
        prot_len_mean=m([p["prot_len"] for p in valid]),
        non_acgt_frac_mean=m([p.get("non_acgt_frac", 0.0) for p in per]),
    )
