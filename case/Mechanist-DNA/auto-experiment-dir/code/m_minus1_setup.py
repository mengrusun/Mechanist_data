"""
M(-1) setup-precondition smoke test.
Verifies: Evo2-7B load + Layer-26 residual extraction + SAE round-trip + DSSP.
Writes results/setup_report.json.
Run: CUDA_VISIBLE_DEVICES=2 python code/m_minus1_setup.py
"""
import os, sys, json, subprocess, traceback, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from evo2_sae import load_evo2, BatchTopKSAE, resolve_sae_layer, tokenize_seq, HIDDEN, SAE_DICT, SAE_K

OUT = "/data/wanghaoxiong/intergene_mechanist_v6/results/setup_report.json"
OMEGAFOLD_WEIGHTS_PATH = os.path.expanduser("~/.cache/omegafold_ckpt/model.pt")
FROZEN_S = "/data/wanghaoxiong/intergene_mechanist_v6/rounds/round_1/results/m0_feature_set.json"
# Crambin (1CRN) real sequence: mixed helix + sheet, 46 aa -> good dual-predictor smoke target.
TEST_PROT = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"
report = {"tool_versions": {}, "smoke_tests": {}, "weight_paths": {}}


def rec(name, ok, detail):
    report["smoke_tests"][name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def main():
    import evo2, vortex, transformers, Bio, sklearn, scipy, statsmodels
    report["tool_versions"] = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "evo2": getattr(evo2, "__version__", "?"),
        "transformers": transformers.__version__,
        "biopython": Bio.__version__, "sklearn": sklearn.__version__,
    }
    report["weight_paths"] = {
        "evo2_7b": "/data1/share_model/evo2/evo2_7b/evo2_7b.pt",
        "sae": "/data1/share_model/evo2/evo2_sae_layer26_mixed/sae-layer26-mixed-expansion_8-k_64.pt",
    }
    report["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")

    # 1. Load Evo2-7B
    t0 = time.time()
    model = load_evo2("cuda:0")
    rec("evo2_load", True, f"loaded Evo2-7B in {time.time()-t0:.1f}s")

    # 2. Resolve layer-26 residual submodule
    valid = resolve_sae_layer(model)
    rec("resolve_sae_layer", len(valid) > 0, f"candidates yielding hidden={HIDDEN}: {valid}")
    if not valid:
        raise RuntimeError("no candidate layer yields 4096-dim residual")
    cand_names = [v[0] for v in valid]

    # 3. Forward several natural-ish sequences, extract residual at ALL candidate sites
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), size=512)) for _ in range(3)]
    sae = BatchTopKSAE(device="cuda:0")

    def fvu_cos_l0(x):
        x_hat, z = sae.roundtrip(x)
        fvu = ((x - x_hat).pow(2).sum() / (x - x.mean(0)).pow(2).sum()).item()
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
        l0 = (z > 0).float().sum(-1).mean().item()
        return fvu, cos, l0, z

    # collect residuals per candidate site
    site_acts = {n: [] for n in cand_names}
    for seq in seqs:
        tok = tokenize_seq(model, seq, "cuda:0")
        with torch.no_grad():
            out = model(tok, return_embeddings=True, layer_names=cand_names)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        emb = out[1]
        for n in cand_names:
            site_acts[n].append(emb[n].float()[0])
    lshape = tuple(logits[0].shape) if isinstance(logits, (tuple, list)) else tuple(logits.shape)
    rec("evo2_forward_512nt", all(site_acts[cand_names[0]][0].shape[-1] == HIDDEN for _ in [0]),
        f"residual shape {tuple(site_acts[cand_names[0]][0].shape)}, logits {lshape}")

    # SAE reconstruction per candidate site -> pick the best-reconstructing = true SAE input site
    site_scores = {}
    for n in cand_names:
        xs = torch.cat(site_acts[n], 0)
        fvu, cos, l0, _ = fvu_cos_l0(xs)
        site_scores[n] = {"explained_var": 1 - fvu, "FVU": fvu, "cosine": cos, "L0": l0}
        print(f"  site {n}: explained_var={1-fvu:.3f} cos={cos:.3f} L0={l0:.1f}", flush=True)
    report["sae_site_selection"] = site_scores
    layer_name = max(site_scores, key=lambda n: site_scores[n]["explained_var"])
    report["chosen_sae_layer"] = layer_name
    rec("sae_site_chosen", True, f"best site '{layer_name}' explained_var={site_scores[layer_name]['explained_var']:.3f}")

    # 4. SAE round-trip on the chosen site
    x = torch.cat(site_acts[layer_name], 0)  # (L, 4096)
    x_hat, z = sae.roundtrip(x)
    fvu = ((x - x_hat).pow(2).sum() / (x - x.mean(0)).pow(2).sum()).item()
    cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
    l0 = (z > 0).float().sum(-1).mean().item()
    # 4b. shuffle-input null (computed first so the round-trip gate can use the margin): the SAE
    #     must reconstruct real activations far better than dimension-shuffled ones.
    x_shuf = x[:, torch.randperm(HIDDEN, device=x.device)]
    xhat_s, _ = sae.roundtrip(x_shuf)
    fvu_s = ((x_shuf - xhat_s).pow(2).sum() / (x_shuf - x_shuf.mean(0)).pow(2).sum()).item()
    report["smoke_tests"]["sae_shuffled_input_FVU"] = fvu_s

    report["smoke_tests"]["sae_roundtrip_metrics"] = {
        "FVU": fvu, "explained_var": 1 - fvu, "mean_cosine": cos,
        "mean_L0": l0, "expected_L0": SAE_K, "dict_size": z.shape[-1],
        "shuffled_input_explained_var": 1 - fvu_s,
        "note": ("smoke probe uses RANDOM DNA (off the natural-CDS manifold the SAE was trained on) "
                 "-> lower absolute explained_var than the ~0.34 round-1 natural-CDS figure is "
                 "expected; validity is asserted by the invariants (dict=32768, L0~64) + a clear "
                 "margin over the dimension-shuffled null. Paper-feature reproduction on natural CDS "
                 "was the decisive validity check (round-1 M(-1))."),
    }
    # Valid-SAE invariants + non-trivial reconstruction vs the shuffled-input null.
    ok_rt = ((z.shape[-1] == SAE_DICT) and (abs(l0 - SAE_K) < 1.5)
             and ((1 - fvu) - (1 - fvu_s) > 0.25) and (cos > 0.5))
    rec("sae_roundtrip", ok_rt,
        f"explained_var={1-fvu:.3f} (vs shuffled {1-fvu_s:.3f}) cosine={cos:.3f} "
        f"L0={l0:.1f} (expect~{SAE_K}) dict={z.shape[-1]}")
    rec("sae_beats_shuffled", (1 - fvu) > (1 - fvu_s) + 0.1,
        f"real explained_var={1-fvu:.3f} vs shuffled-input {1-fvu_s:.3f}")

    del model
    torch.cuda.empty_cache()

    # 5. DSSP (mkdssp) on a tiny synthetic PDB test — download a small real PDB
    dssp_ok, dssp_detail = test_dssp()
    rec("dssp_mkdssp", dssp_ok, dssp_detail)

    # 6. Dual structure predictor: ESMFold (default) + OmegaFold (independent second predictor).
    #    Fold the same test protein with both, run DSSP on each output.
    test_predictors()

    # 7. Frozen round-1 feature set S loads (19 helix + 5 beta + 19 matched-control + s_f).
    test_frozen_s()

    report["chosen_structure_predictor"] = "ESMFold (facebook/esmfold_v1)"
    report["chosen_second_predictor"] = {
        "name": "OmegaFold (release2 / model 2)",
        "why": ("Independent single-sequence predictor with its own OmegaPLM language model (NOT "
                "MSA-based, NOT the ESM2 trunk ESMFold uses) -> the most independent second axis "
                "available without licensed AlphaFold DBs or MSA search. Installed from the repo on "
                "PYTHONPATH (its pip setup.py rejects py3.11); release2 weights (~3.18GB) fetched "
                "directly from helixon S3 (HC4 install-and-continue, no sudo/creds/quota block). "
                "OmegaFold + ESMFold both emit per-residue confidence on a 0-100 scale into the PDB "
                "b-factor, feeding a common pLDDT-weighted DSSP readout."),
        "weights_path": OMEGAFOLD_WEIGHTS_PATH,
        "num_cycle": 4,
    }

    report["overall_pass"] = all(v["pass"] for k, v in report["smoke_tests"].items()
                                 if isinstance(v, dict) and "pass" in v)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", OUT, "overall_pass=", report["overall_pass"], flush=True)


def test_predictors():
    """Fold TEST_PROT with ESMFold and OmegaFold; run DSSP on each; require coords+pLDDT+SS."""
    from fold import esmfold_pdb, omegafold_pdb, structural_readout
    for name, fn in (("esmfold", esmfold_pdb), ("omegafold", omegafold_pdb)):
        try:
            t0 = time.time()
            pdb, plddt = fn(TEST_PROT, device="cuda:0")
            if pdb is None:
                rec(name, False, "fold returned None")
                continue
            rd = structural_readout(pdb)
            dt = time.time() - t0
            if rd is None:
                rec(name, False, f"folded (plddt={plddt:.1f}) but DSSP/readout failed")
                continue
            ok = (plddt is not None) and (rd["n_resolved"] >= 20) and (0.0 <= rd["helix_hgi_w"] <= 1.0)
            rec(name, ok,
                f"folded {len(TEST_PROT)}aa in {dt:.1f}s -> mean_plddt={plddt:.1f}, "
                f"n_resolved={rd['n_resolved']}, helix_hgi={rd['helix_hgi']:.2f}, "
                f"helix_hgi_w(pLDDT-weighted)={rd['helix_hgi_w']:.2f}, sheet={rd['sheet']:.2f}")
            torch.cuda.empty_cache()
        except Exception as e:
            rec(name, False, f"exception: {type(e).__name__}: {str(e)[:200]}\n{traceback.format_exc()[-400:]}")


def test_frozen_s():
    """Confirm the frozen round-1 feature set S loads with the expected structure."""
    try:
        S = json.load(open(FROZEN_S))
        nh = len(S.get("helix_features", []))
        nb = len(S.get("beta_features", []))
        nm = len(S.get("matched_control_features", []))
        ok = (nh >= 15 and len(S.get("s_f", [])) == nh and nb >= 3 and nm >= 15
              and S.get("verdict") == "established")
        rec("frozen_S_loads", ok,
            f"S={nh} helix (+{len(S.get('s_f',[]))} s_f), {nb} beta, {nm} matched-control; "
            f"round-1 verdict={S.get('verdict')}; set-AUROC(prok|HGI|s200)="
            f"{S.get('per_config_combined_set_auroc',{}).get('prokaryote|HGI|s200')}")
    except Exception as e:
        rec("frozen_S_loads", False, f"exception: {type(e).__name__}: {str(e)[:200]}")


def test_dssp():
    """Fetch a small PDB and run mkdssp; confirm per-residue SS returned."""
    try:
        import urllib.request
        pdb_id = "1CRN"  # crambin, 46 residues, well-resolved
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        loc = "/data/wanghaoxiong/intergene_mechanist_v6/data/1crn.pdb"
        if not os.path.exists(loc):
            urllib.request.urlretrieve(url, loc)
        which = subprocess.run(["which", "mkdssp"], capture_output=True, text=True).stdout.strip()
        out_dssp = "/data/wanghaoxiong/intergene_mechanist_v6/data/1crn.dssp"
        # mkdssp v4 CLI: mkdssp input output  (or --output-format dssp)
        r = subprocess.run(["mkdssp", loc, out_dssp], capture_output=True, text=True)
        if r.returncode != 0:
            # try v4 syntax variants
            r = subprocess.run(["mkdssp", "--output-format", "dssp", loc, out_dssp],
                               capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out_dssp):
            return False, f"mkdssp failed rc={r.returncode} err={r.stderr[:200]}"
        # parse SS column via biopython
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import make_dssp_dict
        d, keys = make_dssp_dict(out_dssp)
        ss = "".join(d[k][1] for k in keys)
        helix = sum(c in "HGI" for c in ss)
        return True, f"mkdssp {which}; {len(ss)} residues; SS='{ss[:40]}...'; helix_frac={helix/len(ss):.2f}"
    except Exception as e:
        return False, f"exception: {type(e).__name__}: {str(e)[:200]}\n{traceback.format_exc()[-400:]}"


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        report["fatal_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        with open(OUT, "w") as f:
            json.dump(report, f, indent=2)
        print("FATAL:", report["fatal_error"], flush=True)
        sys.exit(1)
