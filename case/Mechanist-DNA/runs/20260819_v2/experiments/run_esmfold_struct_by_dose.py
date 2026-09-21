"""
ESMFold structural %H + pLDDT by dose, with a mkdssp-4.x-correct DSSP parse.
Folds N seqs/dose for all doses, computes:
  - mean pLDDT (CA B-factor, rescaled to 0-100)
  - structural DSSP %H (H,G,I) and %E (E,B) from the folded model  [the piece that was nan before]
Writes runs/followup_esmfold/struct_by_dose.json and prints a table.
"""
import os, sys, json, subprocess, tempfile
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import evo2lib as E

ESMFOLD_PATH = "/mnt/quarkfs/share_model/esmfold_v1"
N_PER_ALPHA = 12; MIN_AA = 20; MAX_AA = 400
OUT_DIR = "runs/followup_esmfold"; os.makedirs(OUT_DIR, exist_ok=True)
DSSP3 = {'H':'H','G':'H','I':'H','E':'E','B':'E','T':'C','S':'C','P':'C','-':'C',' ':'C','C':'C'}

def ca_plddt(pdb):
    v=[float(l[60:66]) for l in pdb.splitlines() if l.startswith("ATOM") and l[12:16].strip()=="CA"]
    return 100.0*float(np.mean(v)) if v else float("nan")   # this build writes pLDDT on 0-1 -> *100

def struct_ss(pdb_str):
    """mkdssp 4.x legacy-format SS parse -> (%H, %E). nan on failure."""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
            f.write(pdb_str); pin=f.name
        pout=pin+".dssp"
        r=subprocess.run(["mkdssp","--output-format","dssp",pin,pout],capture_output=True,text=True)
        if not os.path.exists(pout) or os.path.getsize(pout)==0:
            subprocess.run(["mkdssp",pin,pout],capture_output=True,text=True)
        if not os.path.exists(pout) or os.path.getsize(pout)==0:
            return float("nan"), float("nan")
        lines=open(pout).read().splitlines()
        start=None
        for i,l in enumerate(lines):
            if l.startswith("  #  RESIDUE"): start=i+1; break
        if start is None: return float("nan"), float("nan")
        ss=[]
        for l in lines[start:]:
            if len(l)<17: continue
            if l[13]=='!': continue
            ss.append(l[16] if l[16]!=' ' else 'C')
        for p in (pin,pout):
            try: os.unlink(p)
            except Exception: pass
        if not ss: return float("nan"), float("nan")
        H=sum(DSSP3.get(c,'C')=='H' for c in ss); Ee=sum(DSSP3.get(c,'C')=='E' for c in ss)
        return 100.0*H/len(ss), 100.0*Ee/len(ss)
    except Exception:
        return float("nan"), float("nan")

def collect():
    gens=json.load(open("results/M2_generations.json"))
    alphas=sorted({float(k.split("_")[0][1:]) for k in gens})
    by={}
    for a in alphas:
        cells=sorted(k for k in gens if abs(float(k.split("_")[0][1:])-a)<1e-9)
        pools=[gens[c] for c in cells]; prots=[]
        for i in range(max((len(p) for p in pools),default=0)):
            for p in pools:
                if i<len(p):
                    pr,fr,st,orf=E.find_longest_orf(p[i],min_aa=MIN_AA)
                    if pr and len(pr)>=MIN_AA: prots.append(pr[:MAX_AA])
                if len(prots)>=N_PER_ALPHA: break
            if len(prots)>=N_PER_ALPHA: break
        by[a]=prots
    return alphas, by

def main():
    from transformers import AutoTokenizer, EsmForProteinFolding
    print("loading ESMFold...", flush=True)
    tok=AutoTokenizer.from_pretrained(ESMFOLD_PATH)
    model=EsmForProteinFolding.from_pretrained(ESMFOLD_PATH, low_cpu_mem_usage=True).cuda().eval()
    model.esm=model.esm.half()
    try: model.trunk.set_chunk_size(64)
    except Exception: pass
    alphas, by = collect()
    print("counts:", {a:len(by[a]) for a in alphas}, flush=True)
    res={}
    for a in alphas:
        pl,H,Ee,ln=[],[],[],[]
        for seq in by[a]:
            try:
                with torch.no_grad(): pdb=model.infer_pdb(seq)
            except Exception as ex:
                print(f"  fold fail a={a}: {ex}", flush=True); continue
            pl.append(ca_plddt(pdb)); h,e=struct_ss(pdb); H.append(h); Ee.append(e); ln.append(len(seq))
        m=lambda x:float(np.nanmean(x)) if x else float("nan")
        sd=lambda x:float(np.nanstd(x)) if x else float("nan")
        res[a]={"n":len(pl),"mean_plddt":m(pl),"std_plddt":sd(pl),
                "struct_pctH":m(H),"struct_pctH_sd":sd(H),"struct_pctE":m(Ee),"mean_len":m(ln)}
        print(f"a={a:>5} n={res[a]['n']:>2} pLDDT={res[a]['mean_plddt']:5.1f}"
              f" struct%H={res[a]['struct_pctH']:5.1f}±{res[a]['struct_pctH_sd']:4.1f}"
              f" struct%E={res[a]['struct_pctE']:5.1f} len={res[a]['mean_len']:5.1f}", flush=True)
    json.dump({"milestone":"followup_esmfold_struct_by_dose","by_alpha":{str(a):res[a] for a in alphas}},
              open(os.path.join(OUT_DIR,"struct_by_dose.json"),"w"), indent=1)
    print("DONE", flush=True)

if __name__=="__main__": main()
