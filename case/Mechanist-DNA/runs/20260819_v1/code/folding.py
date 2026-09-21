"""ESMFold -> DSSP secondary-structure readout.

Primary path: transformers EsmForProteinFolding ("facebook/esmfold_v1").
Produces a PDB string per protein; DSSP (mkdssp) -> 3-state SS -> %helix.
Reports helix fraction over all residues and over pLDDT>=70 residues.
"""
import os, sys, tempfile, io
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import common as C


ESMFOLD_LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data/esmfold_model")

class ESMFolder:
    def __init__(self, device="cuda:0", chunk_size=64):
        from transformers import AutoTokenizer, EsmForProteinFolding
        src = ESMFOLD_LOCAL if os.path.exists(
            os.path.join(ESMFOLD_LOCAL, "pytorch_model.bin")) else "facebook/esmfold_v1"
        self.tok = AutoTokenizer.from_pretrained(src)
        self.model = EsmForProteinFolding.from_pretrained(src, low_cpu_mem_usage=True)
        self.model = self.model.to(device).eval()
        self.device = device
        try:
            self.model.trunk.set_chunk_size(chunk_size)
        except Exception:
            pass

    @torch.no_grad()
    def fold_pdb(self, aa_seq):
        """Return (pdb_str, mean_plddt, per_res_plddt np.array)."""
        inp = self.tok([aa_seq], return_tensors="pt", add_special_tokens=False)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.model(**inp)
        pdb = self.model.output_to_pdb(out)[0]
        # per-residue pLDDT: plddt (B,L,37) atom-level -> mean over atoms present
        plddt = out["plddt"][0].cpu().numpy()        # (L,37)
        atom_mask = out["atom37_atom_exists"][0].cpu().numpy()
        per_res = (plddt * atom_mask).sum(-1) / np.clip(atom_mask.sum(-1), 1, None)
        return pdb, float(per_res.mean()), per_res

    def helix_readout(self, aa_seq):
        """Fold + DSSP. Returns dict with %helix (all & plddt>=70), %sheet, %coil,
        mean pLDDT, length, ss3 string."""
        pdb, mean_plddt, per_res = self.fold_pdb(aa_seq)
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as fh:
            fh.write(pdb); path = fh.name
        try:
            ss8 = C.run_dssp_on_pdb(path)
        finally:
            os.unlink(path)
        ss3 = C.three_state(ss8) if ss8 else ""
        # align per_res length to ss3 (DSSP may drop residues); use min length
        if ss3 and len(per_res) >= len(ss3):
            plddt_ss = per_res[:len(ss3)]
        else:
            plddt_ss = per_res
        hi = np.array([c for c, p in zip(ss3, plddt_ss) if p >= 70.0]) if ss3 else np.array([])
        def frac(arr, state):
            if len(arr) == 0:
                return float("nan")
            return float(np.mean([c == state for c in arr]))
        ss3_arr = np.array(list(ss3)) if ss3 else np.array([])
        return dict(
            helix_all=frac(ss3_arr, "H"), sheet_all=frac(ss3_arr, "E"),
            coil_all=frac(ss3_arr, "C"),
            helix_hi=frac(hi, "H") if len(hi) else float("nan"),
            n_hi=int(len(hi)), mean_plddt=mean_plddt, ss_len=len(ss3),
            ss3=ss3,
        )


if __name__ == "__main__":
    # self-test on a known helical protein fragment
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    import time
    t = time.time()
    f = ESMFolder(device=dev)
    print(f"[fold] loaded ESMFold in {time.time()-t:.0f}s", flush=True)
    # myoglobin-like helical seq (should be helix-rich)
    seq = ("MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRVKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGH"
           "HEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG")
    t = time.time()
    r = f.helix_readout(seq)
    print(f"[fold] folded {len(seq)}aa in {time.time()-t:.1f}s: "
          f"helix_all={r['helix_all']:.2f} helix_hi={r['helix_hi']:.2f} "
          f"sheet={r['sheet_all']:.2f} plddt={r['mean_plddt']:.1f} ss_len={r['ss_len']}", flush=True)
    print(f"[fold] ss3[:60]={r['ss3'][:60]}", flush=True)
