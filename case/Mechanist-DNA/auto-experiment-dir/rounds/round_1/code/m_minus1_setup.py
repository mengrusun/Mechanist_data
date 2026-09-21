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
    report["smoke_tests"]["sae_roundtrip_metrics"] = {
        "FVU": fvu, "explained_var": 1 - fvu, "mean_cosine": cos,
        "mean_L0": l0, "expected_L0": SAE_K, "dict_size": z.shape[-1],
    }
    # A valid SAE on this activation space should explain a substantial fraction of variance
    ok_rt = (z.shape[-1] == SAE_DICT) and (abs(l0 - SAE_K) < 1.0) and (1 - fvu > 0.5) and (cos > 0.7)
    rec("sae_roundtrip", ok_rt,
        f"explained_var={1-fvu:.3f} cosine={cos:.3f} L0={l0:.1f} (expect~{SAE_K}) dict={z.shape[-1]}")

    # 4b. shuffle-input null: SAE should reconstruct real activations far better than shuffled ones
    x_shuf = x[:, torch.randperm(HIDDEN, device=x.device)]
    xhat_s, _ = sae.roundtrip(x_shuf)
    fvu_s = ((x_shuf - xhat_s).pow(2).sum() / (x_shuf - x_shuf.mean(0)).pow(2).sum()).item()
    report["smoke_tests"]["sae_shuffled_input_FVU"] = fvu_s
    rec("sae_beats_shuffled", (1 - fvu) > (1 - fvu_s) + 0.1,
        f"real explained_var={1-fvu:.3f} vs shuffled-input {1-fvu_s:.3f}")

    del model
    torch.cuda.empty_cache()

    # 5. DSSP (mkdssp) on a tiny synthetic PDB test — download a small real PDB
    dssp_ok, dssp_detail = test_dssp()
    rec("dssp_mkdssp", dssp_ok, dssp_detail)

    report["overall_pass"] = all(v["pass"] for k, v in report["smoke_tests"].items()
                                 if isinstance(v, dict) and "pass" in v)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", OUT, "overall_pass=", report["overall_pass"], flush=True)


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
