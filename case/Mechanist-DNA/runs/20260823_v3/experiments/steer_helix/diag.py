"""One-shot diagnostic: fix DSSP + probe SAE reconstruction convention."""
import os, sys, subprocess, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import common as C

os.makedirs("/data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/results/diag", exist_ok=True)
DIAG = "/data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/results/diag"

# ---------- 1. ESMFold a helix, dump PDB ----------
assay = C.HelixAssay()
helix_pep = "MEEELKKLLEELKKLGSSEEELKKLLEELKKLGSSEEELKKLLEELKKLG"
pdb, plddt = assay.fold_to_pdb(helix_pep)
pdb_path = os.path.join(DIAG, "helix.pdb")
open(pdb_path, "w").write(pdb)
print("PDB written, plddt=", plddt, "n_lines=", len(pdb.splitlines()))
print("PDB head:\n", "\n".join(pdb.splitlines()[:3]))

# ---------- 2. Test DSSP binaries directly ----------
for name, binp in [("usr_mkdssp", "/usr/bin/mkdssp"),
                   ("conda_mkdssp", "/data/wanghaoxiong/miniconda3/envs/scientist/bin/mkdssp")]:
    for fmt in [None, "dssp"]:
        outp = os.path.join(DIAG, f"{name}_{fmt}.out")
        cmd = [binp]
        if fmt: cmd += ["--output-format", fmt]
        cmd += [pdb_path, outp]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            head = open(outp).read()[:200] if os.path.exists(outp) else "<no file>"
            print(f"\n[{name} fmt={fmt}] rc={r.returncode} stderr={r.stderr[:150]!r}")
            print("  out head:", head[:120].replace("\n", " | "))
        except Exception as e:
            print(f"\n[{name} fmt={fmt}] EXC {e}")

# ---------- 3. BioPython DSSP with conda binary ----------
from Bio.PDB import PDBParser, DSSP
for binp in ["/data/wanghaoxiong/miniconda3/envs/scientist/bin/mkdssp", "/usr/bin/mkdssp"]:
    try:
        s = PDBParser(QUIET=True).get_structure("x", pdb_path)
        d = DSSP(s[0], pdb_path, dssp=binp)
        ss = [d[k][2] for k in d.keys()]
        helix = sum(c in ("H", "G", "I") for c in ss)
        print(f"\n[BioDSSP {os.path.basename(binp)}] n_res={len(ss)} helix_frac={helix/max(1,len(ss)):.3f} ss={''.join(ss)[:40]}")
    except Exception as e:
        print(f"\n[BioDSSP {os.path.basename(binp)}] EXC {str(e)[:150]}")

# ---------- 4. SAE reconstruction on REAL per-token block-26 activations ----------
evo = C.Evo2Wrapper("evo2_7b")
seqs = evo.generate(C.DNA_PRIMERS[:2], n_tokens=180, seed=0)
# per-token acts at block 26
ids = torch.tensor(evo.tokenizer.tokenize(C._clean_dna(seqs[0])), dtype=torch.long)[None].cuda()
_, emb = evo.evo2.forward(ids, return_embeddings=True, layer_names=["blocks.26"])
A = emb["blocks.26"]
if isinstance(A, tuple): A = A[0]
A = A[0].float()  # (seq, 4096)
print("\nper-token act block26 shape", tuple(A.shape), "mean_norm", A.norm(dim=-1).mean().item())

sd = torch.load(C.SAE_PATH, map_location="cuda")
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
W = sd["W"].float()           # (4096, 32768)
b_enc = sd["b_enc"].float()   # (32768,)
b_dec = sd["b_dec"].float()   # (4096,)
print("W", tuple(W.shape), "b_enc", tuple(b_enc.shape), "b_dec", tuple(b_dec.shape))

def topk(f, k=64):
    v, i = torch.topk(f, k, dim=-1)
    return torch.zeros_like(f).scatter_(-1, i, v)

def rel(x, xh): return (torch.norm(x - xh) / (torch.norm(x) + 1e-9)).item()

x = A
convs = {}
# conv A: encode (x-b_dec)@W + b_enc ; decode f@W.T + b_dec
f = topk(torch.relu((x - b_dec) @ W + b_enc)); convs["A_sub_bdec_WT"] = rel(x, f @ W.T + b_dec)
# conv B: encode x@W + b_enc ; decode f@W.T + b_dec
f = topk(torch.relu(x @ W + b_enc)); convs["B_nosub_WT"] = rel(x, f @ W.T + b_dec)
# conv C: encode (x-b_dec)@W + b_enc ; decode f@W.T (no bias add) + b_dec already; try decode f@W.T only
f = topk(torch.relu((x - b_dec) @ W + b_enc)); convs["C_sub_WT_nobdec_out"] = rel(x, f @ W.T)
# conv D: W as (n_feat,d_model)? W is (4096,32768) so W.T is (32768,4096); encode x@ W? already. try decode f@W where W(4096,32768)->no. skip
# conv E: no topk (dense) with sub
f = torch.relu((x - b_dec) @ W + b_enc); convs["E_dense_sub"] = rel(x, f @ W.T + b_dec)
# active count
fk = topk(torch.relu((x - b_dec) @ W + b_enc))
print("active per token (topk):", int((fk[0] > 0).sum()))
for k, v in convs.items():
    print(f"  recon[{k}] = {v:.4f}")
print("DIAG DONE")
