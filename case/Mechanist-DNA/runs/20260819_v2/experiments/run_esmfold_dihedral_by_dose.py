"""
ESMFold pLDDT + backbone-dihedral (Ramachandran) helix fraction, by dose.
Self-contained SS estimate from folded N/CA/C coords (no DSSP/biotite needed).
Helix = residues in the alpha/3-10 phi-psi basin; Sheet = beta basin.
NOTE: this is a phi/psi geometric estimate, NOT the DSSP H-bond definition.
Writes runs/followup_esmfold/dihedral_by_dose.json and prints a table.
"""
import os, sys, json
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import evo2lib as E

ESMFOLD_PATH="/mnt/quarkfs/share_model/esmfold_v1"
N_PER_ALPHA=12; MIN_AA=20; MAX_AA=400
OUT="runs/followup_esmfold"; os.makedirs(OUT, exist_ok=True)

def ca_plddt(pdb):
    v=[float(l[60:66]) for l in pdb.splitlines() if l.startswith("ATOM") and l[12:16].strip()=="CA"]
    return 100.0*float(np.mean(v)) if v else float("nan")   # 0-1 build -> *100

def dihedral(p0,p1,p2,p3):
    b0=p0-p1; b1=p2-p1; b2=p3-p2
    b1=b1/ (np.linalg.norm(b1)+1e-9)
    v=b0-np.dot(b0,b1)*b1; w=b2-np.dot(b2,b1)*b1
    x=np.dot(v,w); y=np.dot(np.cross(b1,v),w)
    return np.degrees(np.arctan2(y,x))

def ss_frac_from_pdb(pdb):
    """(%H,%E) from phi/psi basins over the first chain."""
    res={}  # resseq -> {N,CA,C}
    order=[]
    for l in pdb.splitlines():
        if not l.startswith("ATOM"): continue
        atom=l[12:16].strip()
        if atom not in ("N","CA","C"): continue
        rs=int(l[22:26])
        xyz=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
        if rs not in res: res[rs]={}; order.append(rs)
        res[rs][atom]=xyz
    order=sorted(order)
    H=Eb=tot=0
    for i in range(1,len(order)-1):
        rp,rc,rn=res[order[i-1]],res[order[i]],res[order[i+1]]
        if not all(k in rp for k in ("C",)) or not all(k in rc for k in ("N","CA","C")) or "N" not in rn:
            continue
        phi=dihedral(rp["C"],rc["N"],rc["CA"],rc["C"])
        psi=dihedral(rc["N"],rc["CA"],rc["C"],rn["N"])
        tot+=1
        if (-140<=phi<=-30) and (-80<=psi<=30): H+=1          # alpha/3-10 basin
        elif (-180<=phi<=-40) and (80<=psi<=180): Eb+=1        # beta basin
    if tot==0: return float("nan"), float("nan")
    return 100.0*H/tot, 100.0*Eb/tot

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
                print(f" fold fail a={a}: {ex}", flush=True); continue
            pl.append(ca_plddt(pdb)); h,e=ss_frac_from_pdb(pdb); H.append(h); Ee.append(e); ln.append(len(seq))
        m=lambda x:float(np.nanmean(x)) if len(x) else float("nan")
        sd=lambda x:float(np.nanstd(x)) if len(x) else float("nan")
        res[a]={"n":len(pl),"mean_plddt":m(pl),"std_plddt":sd(pl),
                "helix_frac":m(H),"helix_frac_sd":sd(H),"sheet_frac":m(Ee),"mean_len":m(ln)}
        print(f"a={a:>5} n={res[a]['n']:>2} pLDDT={res[a]['mean_plddt']:5.1f}"
              f" helix%={res[a]['helix_frac']:5.1f}±{res[a]['helix_frac_sd']:4.1f}"
              f" sheet%={res[a]['sheet_frac']:5.1f} len={res[a]['mean_len']:5.1f}", flush=True)
    json.dump({"milestone":"followup_esmfold_dihedral_by_dose",
               "ss_method":"phi/psi Ramachandran basin (NOT DSSP H-bond)",
               "by_alpha":{str(a):res[a] for a in alphas}},
              open(os.path.join(OUT,"dihedral_by_dose.json"),"w"), indent=1)
    print("DONE", flush=True)

if __name__=="__main__": main()
