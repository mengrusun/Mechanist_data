"""Probe: build a few natural-CDS+DSSP examples, then re-validate SAE site on natural DNA."""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import m0_data as D
from evo2_sae import load_evo2, BatchTopKSAE, tokenize_seq, HIDDEN

def build_examples(organism, n_target=6):
    cfg = D.ORGANISMS[organism]
    prots = D.uniprot_structured_proteins_paged(cfg["taxid"], max_proteins=40, page_size=40)
    print(f"[{organism}] uniprot structured proteins fetched: {len(prots)}", flush=True)
    examples = []
    for p in prots:
        if len(examples) >= n_target:
            break
        if len(p["seq"]) > 700 or len(p["seq"]) < 60:
            continue
        cds, pid = D.fetch_cds_for_protein(p["embl"], p["seq"], cfg["transl_table"])
        if not cds:
            continue
        # pick a PDB, prefer X-ray small
        got = None
        for pdb in p["pdbs"][:6]:
            loc = D.download_pdb(pdb["id"])
            if not loc:
                continue
            chains = D.run_dssp(loc)
            if not chains:
                continue
            codon_ss, cov, chain = D.align_ss_to_cds(cds, p["seq"], chains, cfg["transl_table"])
            if codon_ss and cov >= 0.5:
                got = {"acc": p["acc"], "pdb": pdb["id"], "chain": chain, "cov": cov,
                       "cds": cds, "codon_ss": codon_ss, "n_codons": len(p["seq"])}
                break
        if got:
            helix = sum(D.helix_label(s, "HGI") == 1 for s in got["codon_ss"])
            labeled = sum(s is not None for s in got["codon_ss"])
            print(f"  {got['acc']} pdb={got['pdb']} chain={got['chain']} cov={cov:.2f} "
                  f"labeled={labeled} helixHGI={helix} ({helix/max(labeled,1):.2f})", flush=True)
            examples.append(got)
    return examples


def main():
    t0 = time.time()
    exs = build_examples("prokaryote", n_target=6)
    print(f"built {len(exs)} prokaryote examples in {time.time()-t0:.0f}s", flush=True)
    if len(exs) < 3:
        print("TOO FEW EXAMPLES — data pipeline needs work"); return
    # SAE site validation on natural CDS activations
    model = load_evo2("cuda:0")
    sae = BatchTopKSAE(device="cuda:0")
    cand = ["blocks.26.mlp.l3", "blocks.26", "blocks.26.pre_norm", "blocks.26.post_norm"]
    site_acts = {c: [] for c in cand}
    for ex in exs:
        tok = tokenize_seq(model, ex["cds"], "cuda:0")
        with torch.no_grad():
            out = model(tok, return_embeddings=True, layer_names=cand)
        emb = out[1]
        for c in cand:
            site_acts[c].append(emb[c].float()[0])
    print("\n=== SAE reconstruction on NATURAL CDS ===", flush=True)
    results = {}
    for c in cand:
        x = torch.cat(site_acts[c], 0)
        x_hat, z = sae.roundtrip(x)
        fvu = ((x - x_hat).pow(2).sum() / (x - x.mean(0)).pow(2).sum()).item()
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
        nmse = ((x - x_hat).pow(2).sum() / x.pow(2).sum()).item()
        l0 = (z > 0).float().sum(-1).mean().item()
        results[c] = {"explained_var": 1-fvu, "cosine": cos, "nmse": nmse, "L0": l0}
        print(f"  {c:22s} explained_var={1-fvu:+.3f} cos={cos:.3f} nmse={nmse:.3f} L0={l0:.1f}", flush=True)
    best = max(results, key=lambda c: results[c]["explained_var"])
    print(f"\nBEST SITE (natural): {best}  explained_var={results[best]['explained_var']:.3f}", flush=True)
    json.dump({"examples": [{k: v for k, v in e.items() if k != 'cds' and k != 'codon_ss'} for e in exs],
               "sae_site_natural": results, "best_site": best},
              open("/data/wanghaoxiong/intergene_mechanist_v6/results/sae_site_natural.json", "w"), indent=2)

if __name__ == "__main__":
    main()
