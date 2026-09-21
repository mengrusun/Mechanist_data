"""Model-axis verify variant for C1 (helix gain / dose-response).

Swap: main experiment model evo2_7b (1M-context checkpoint, config evo2-7b-1m.yml)
  ->  evo2_7b_262k (262k-context checkpoint, config evo2-7b-262k.yml).
Same 7B StripedHyena architecture (32 blocks, hidden 4096) -> WITHIN-FAMILY model swap.

EVERYTHING ELSE FROZEN (minimum diff from experiments/steer_helix/run_m3.py):
  - same CAA method (diff-of-means high-helix minus low-helix at block 28), rebuilt in the
    262k model's OWN activation space over the SAME M1 DEV contrast DNA (model-independent data);
  - same frozen ESMFold->DSSP helix assay (structure predictor is model-independent);
  - same deterministic longest-ATG ORF rule, same validity composite, same primer set;
  - same anti-circularity: CAA built on DEV contrast, evaluated on TEST-split primers with the
    identical TEST seed offset (900000 + seed*1000 + call) used by the main experiment.

Scoped small to fit the verify GPU budget (<= ~8 GPU-h; ~12.7 GPU-h remained this round):
  coefs [0.0, 1.0, 2.0, 4.0] x seeds [0, 1] x n=150  (vs main 6 coef x 3 seed x 250).
  coef 0.0 = in-model baseline control; coef 1.0 = the main experiment's winning setting.
"""
import os, sys, json, time, re
import numpy as np
PROJ = "/data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1"
sys.path.insert(0, os.path.join(PROJ, "experiments/steer_helix"))
import common as C

ALT_PT = "/mnt/quarkfs/share_models/evo2_7b_262k/evo2_7b_262k.pt"
ALT_NAME = "evo2_7b_262k"
SITE = 28
COEFS = [0.0, 1.0, 2.0, 4.0]
SEEDS = [0, 1]
N = 150
OUTDIR = os.path.join(PROJ, "verify/C1_helix_gain_doseresponse/variants/model-swap-evo2-7b-262k")


class Evo2Wrapper262k(C.Evo2Wrapper):
    """Same wrapper as the main experiment, but loads the alternate 262k checkpoint + config."""
    def __init__(self):
        from evo2 import Evo2
        self.evo2 = Evo2(ALT_NAME, local_path=ALT_PT)
        self.model = self.evo2.model
        self.tokenizer = self.evo2.tokenizer
        self._hooks = []
        BLOCK_RE = re.compile(r"^blocks\.(\d+)$")
        self.block_names = sorted(
            [n for n, _ in self.model.named_modules() if BLOCK_RE.match(n)],
            key=lambda n: int(BLOCK_RE.match(n).group(1)))
        self.hidden_size = 4096


def build_caa_vector(evo, block):
    recs = {r["gid"]: r for r in json.load(open(os.path.join(PROJ, "results/m1/baseline_records.json")))}
    con = json.load(open(os.path.join(PROJ, "results/m1/contrast_set.json")))
    hi = [recs[g]["dna"] for g in con["high_gids"]]
    lo = [recs[g]["dna"] for g in con["low_gids"]]
    a_hi = evo.block_activations(hi, [block])[block].mean(0)
    a_lo = evo.block_activations(lo, [block])[block].mean(0)
    return (a_hi - a_lo).astype(np.float32), con["n_pairs"]


def score_generations(evo, assay, seqs, nlls):
    out = []
    for dna, nll in zip(seqs, nlls):
        orf_dna, protein = C.extract_orf(dna)
        ss = assay.ss_fractions(protein) if protein else dict(
            helix_frac=0.0, sheet_frac=0.0, plddt=0.0, n_res=0, ok=False)
        valid, _ = C.validity_label(dna, protein, nll)
        out.append(dict(helix_frac=ss["helix_frac"], sheet_frac=ss["sheet_frac"],
                        plddt=ss["plddt"], mean_nll=nll, gc=C.gc_content(dna),
                        prot_len=(len(protein) if protein else 0),
                        aa_helixfav=C.aa_composition(protein) if protein else 0.0,
                        valid=bool(valid), fold_ok=ss["ok"]))
    return out


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    evo = Evo2Wrapper262k()
    assay = C.HelixAssay()
    vec, n_pairs = build_caa_vector(evo, SITE)
    vnorm = float(np.linalg.norm(vec))
    print(f"[variant-262k] CAA vector built at site {SITE}: vnorm={vnorm:.2f} n_pairs={n_pairs}", flush=True)

    rows = []
    for coef in COEFS:
        for seed in SEEDS:
            n_calls = int(np.ceil(N / len(C.DNA_PRIMERS)))
            gen = []
            for cc in range(n_calls):
                evo.clear_hooks()
                if coef != 0.0:
                    evo.add_steering_hook(SITE, vec, coef)
                seqs = evo.generate(C.DNA_PRIMERS, seed=900000 + seed * 1000 + cc)  # TEST offset
                evo.clear_hooks()
                nlls = evo.sequence_nll(seqs)
                gen.extend(score_generations(evo, assay, seqs, nlls))
            gen = gen[:N]
            hall = np.array([g["helix_frac"] for g in gen])
            vrate = float(np.mean([g["valid"] for g in gen]))
            rows.append(dict(site=SITE, mode="caa", coef=coef, seed=seed, n=len(gen),
                             helix_all=float(hall.mean()),
                             helix_valid=float(np.mean([g["helix_frac"] for g in gen if g["valid"]])
                                               if any(g["valid"] for g in gen) else 0.0),
                             validity_rate=vrate,
                             mean_nll=float(np.nanmean([g["mean_nll"] for g in gen])),
                             sheet=float(np.mean([g["sheet_frac"] for g in gen])),
                             gc=float(np.mean([g["gc"] for g in gen])),
                             plddt=float(np.mean([g["plddt"] for g in gen])),
                             prot_len=float(np.mean([g["prot_len"] for g in gen])),
                             aa_helixfav=float(np.mean([g["aa_helixfav"] for g in gen])),
                             per_gen_helix=[float(x) for x in hall]))
            print("[variant-262k] coef=%.2f seed=%d helix_all=%.3f valid=%.3f nll=%.3f len=%.1f"
                  % (coef, seed, hall.mean(), vrate, np.nanmean([g["mean_nll"] for g in gen]),
                     np.mean([g["prot_len"] for g in gen])), flush=True)

    result = dict(model=ALT_NAME, local_path=ALT_PT, swap_axis="model",
                  main_experiment_model="evo2_7b", site=SITE, mode="caa",
                  vnorm=vnorm, n_pairs=n_pairs, coefs=COEFS, seeds=SEEDS, n_per_cell=N,
                  wall_seconds=round(time.time() - t0, 1), rows=rows)
    C.save_json(result, os.path.join(OUTDIR, "result.json"))
    print(f"[variant-262k] DONE wall={result['wall_seconds']}s -> {OUTDIR}/result.json", flush=True)


if __name__ == "__main__":
    main()
