"""Sanity smoke test (smallest, fastest): exercises the full stack end-to-end at tiny scale.
Loads Evo2-7B, generates a few short sequences, extracts ORFs, folds 1-2 with ESMFold->DSSP,
builds a trivial steering vector and confirms a steered generation runs + differs. NOT a result."""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--out", default=os.path.join(C.PROJECT, "results/sanity"))
    args = ap.parse_args()
    if args.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)
    report = {}

    evo = C.Evo2Wrapper("evo2_7b")
    report["blocks_found"] = len(evo.block_names)
    seqs = evo.generate(C.DNA_PRIMERS[:2], n_tokens=180, seed=0)
    report["gen_ok"] = all(len(s) > 10 for s in seqs)
    report["example_len"] = [len(s) for s in seqs]

    orf_dna, protein = C.extract_orf(seqs[0])
    report["orf_len_aa"] = (len(protein) if protein else 0)

    assay = C.HelixAssay()
    if protein and len(protein) >= 10:
        ss = assay.ss_fractions(protein[:120])
        report["assay"] = ss
    else:
        ss2 = assay.ss_fractions("MEEELKKLLEELKKLGSSEEELKKLLEELKKLG")  # synthetic helix probe
        report["assay_synthetic"] = ss2

    # activation extraction + steering hook smoke
    acts = evo.block_activations(seqs, [26])[26]
    report["act_shape"] = list(acts.shape)
    vec = (acts[0] - acts[-1]).astype(np.float32) if len(acts) > 1 else acts[0]
    evo.clear_hooks(); evo.add_steering_hook(26, vec, 2.0)
    steered = evo.generate(C.DNA_PRIMERS[:2], n_tokens=180, seed=0)
    evo.clear_hooks()
    report["steer_changes_output"] = bool(steered[0] != seqs[0])
    report["nll_ok"] = bool(np.isfinite(evo.sequence_nll(seqs[:1])[0]))

    # SAE load smoke
    try:
        sae = C.TopKSAE(); import torch
        A = torch.tensor(acts[:min(4, len(acts))]).cuda()
        f = sae.encode(A)
        report["sae_encode_shape"] = list(f.shape)
        report["sae_n_features"] = int(sae.n_features)
        report["sae_active_per_row"] = int((f[0] > 0).sum().item())   # should be ~k=64
        report["sae_recon_rel_error"] = sae.reconstruction_rel_error(A)  # small => orientation OK
        report["sae_recon_rel_error_norm"] = sae.reconstruction_rel_error_normalized(A)
    except Exception as e:
        report["sae_error"] = str(e)[:200]

    C.save_json(report, os.path.join(args.out, "sanity_report.json"))
    ok = report.get("gen_ok") and report.get("steer_changes_output")
    print("[SANITY]", json.dumps(report, indent=2, default=float))
    print("[SANITY] PASS" if ok else "[SANITY] FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
