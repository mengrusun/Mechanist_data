#!/usr/bin/env python3
"""Finalize analysis for the Evo2-7B helix-steering experiment.
Computes pre-registered C1 (dose-response trend) and C2 (validity non-inferiority)
stats from on-disk raw files. No generation. Emits specificity.json + validity_frontier.json."""
import json, math, random
import numpy as np

random.seed(0); np.random.seed(0)
R = "/data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/results"

def load(p):
    with open(p) as f: return json.load(f)

# ---------- helpers ----------
def cliffs_delta(a, b):
    a=np.asarray(a); b=np.asarray(b)
    # P(a>b)-P(a<b); use rank-based O(n log n)
    gt=lt=0
    # vectorized via sorting
    combined=np.concatenate([a,b])
    order=np.argsort(combined, kind="mergesort")
    # simpler: since n=750 each, direct broadcast (750x750=562k) is fine
    diff=a[:,None]-b[None,:]
    gt=np.sum(diff>0); lt=np.sum(diff<0)
    return (gt-lt)/(len(a)*len(b))

def hedges_g(a, b):
    a=np.asarray(a); b=np.asarray(b)
    n1,n2=len(a),len(b)
    s1,s2=a.std(ddof=1),b.std(ddof=1)
    sp=math.sqrt(((n1-1)*s1**2+(n2-1)*s2**2)/(n1+n2-2))
    d=(a.mean()-b.mean())/sp
    J=1-3/(4*(n1+n2)-9)
    return d*J

def mannwhitney_p(a, b):
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        # normal-approx fallback
        a=np.asarray(a); b=np.asarray(b); n1,n2=len(a),len(b)
        allv=np.concatenate([a,b]); ranks=allv.argsort().argsort()+1
        R1=ranks[:n1].sum(); U1=R1-n1*(n1+1)/2; U=min(U1,n1*n2-U1)
        mu=n1*n2/2; sd=math.sqrt(n1*n2*(n1+n2+1)/12)
        z=(U-mu)/sd; from math import erf
        return 2*(1-0.5*(1+erf(abs(z)/math.sqrt(2))))

def spearman(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    rx=x.argsort().argsort().astype(float); ry=y.argsort().argsort().astype(float)
    rx-=rx.mean(); ry-=ry.mean()
    return float((rx@ry)/(math.sqrt(rx@rx)*math.sqrt(ry@ry)))

def perm_trend_p(coef, val, nperm=20000):
    """Permutation test: H0 = no monotone assoc between coef and val. Stat=Spearman rho."""
    obs=spearman(coef, val)
    val=np.asarray(val); coef=np.asarray(coef)
    cnt=0
    for _ in range(nperm):
        if spearman(coef, np.random.permutation(val))>=obs: cnt+=1
    return obs, (cnt+1)/(nperm+1)

def boot_ci_diff(a, b, nboot=10000, alpha=0.05):
    a=np.asarray(a); b=np.asarray(b); rng=np.random.default_rng(0)
    diffs=np.empty(nboot)
    for i in range(nboot):
        diffs[i]=rng.choice(a,len(a),replace=True).mean()-rng.choice(b,len(b),replace=True).mean()
    lo=np.percentile(diffs,100*alpha/2); hi=np.percentile(diffs,100*(1-alpha/2))
    lb1=np.percentile(diffs,100*alpha)  # one-sided lower bound
    return float(lo),float(hi),float(lb1)

def bh_fdr(pvals):
    p=np.asarray(pvals); n=len(p); order=p.argsort()
    ranked=p[order]; adj=ranked*n/(np.arange(1,n+1))
    adj=np.minimum.accumulate(adj[::-1])[::-1]
    out=np.empty(n); out[order]=np.clip(adj,0,1)
    return out.tolist()

# ================= C1: dose-response trend (site 28 CAA primary) =================
def site_rows(fn):
    d=load(f"{R}/m3/{fn}")
    rows=d["rows"]
    coef=[r["coef"] for r in rows]; helix=[r["helix_all"] for r in rows]; val=[r["validity_rate"] for r in rows]
    return coef, helix, val, rows

c28,h28,v28,rows28 = site_rows("s28_caa.json")
c30,h30,v30,_ = site_rows("s30_caa.json")
c26,h26,v26,_ = site_rows("s26_sae_clamp.json")

rho28,p28 = perm_trend_p(c28,h28)
rho30,p30 = perm_trend_p(c30,h30)
rho26,p26 = perm_trend_p(c26,h26)
fdr = bh_fdr([p28,p30,p26])

# per-coef means (site 28) for dose ladder
def by_coef(coef,val):
    out={}
    for c,v in zip(coef,val): out.setdefault(c,[]).append(v)
    return {c:float(np.mean(vs)) for c,vs in sorted(out.items())}
dose28_helix=by_coef(c28,h28); dose28_val=by_coef(c28,v28)

# ================= M4 per-gen: winning vs baseline + controls =================
def pg(cond):
    d=load(f"{R}/m4/raw_{cond}.json")
    g=d["per_gen"]
    return {
        "helix":np.array([x["helix_frac"] for x in g]),
        "valid":np.array([1.0 if x["valid"] else 0.0 for x in g]),
        "sheet":np.array([x["sheet_frac"] for x in g]),
        "gc":np.array([x["gc"] for x in g]),
        "nll":np.array([x["mean_nll"] for x in g]),
        "plen":np.array([x["prot_len"] for x in g]),
        "aahf":np.array([x["aa_helixfav"] for x in g]),
        "n":len(g),
    }
base=pg("baseline"); caa=pg("caa_win"); mc=pg("matched_control"); sham=pg("sham")

def mean(d,k): return float(d[k].mean())

# winning vs baseline effect size (per-gen helix, invalid already helix=0 in raw? check)
# raw helix_frac is the folded helix; primary endpoint mean over all gens.
delta_wb = cliffs_delta(caa["helix"], base["helix"])
g_wb = hedges_g(caa["helix"], base["helix"])
lo_wb,hi_wb,_ = boot_ci_diff(caa["helix"], base["helix"])

# ================= Specificity =================
def contrast(name,a,b):
    return {
        "helix_mean_a":float(a["helix"].mean()),
        "helix_mean_b":float(b["helix"].mean()),
        "delta_helix":float(a["helix"].mean()-b["helix"].mean()),
        "cliffs_delta":float(cliffs_delta(a["helix"],b["helix"])),
        "hedges_g":float(hedges_g(a["helix"],b["helix"])),
        "mannwhitney_p":mannwhitney_p(a["helix"],b["helix"]),
    }
spec = {
  "primary_endpoint":"mean_helix_all_generations (invalid->0)",
  "winning_setting":{"site":28,"mode":"caa","coef":1.0},
  "n_per_condition":caa["n"],
  "means":{"baseline":mean(base,"helix"),"caa_win":mean(caa,"helix"),
           "matched_control":mean(mc,"helix"),"sham":mean(sham,"helix")},
  "caa_vs_baseline":contrast("caa_vs_baseline",caa,base),
  "matched_control_vs_baseline":contrast("mc_vs_baseline",mc,base),
  "sham_vs_baseline":contrast("sham_vs_baseline",sham,base),
  "caa_vs_matched_control":contrast("caa_vs_mc",caa,mc),
  "caa_vs_sham":contrast("caa_vs_sham",caa,sham),
}
# specificity multiplicity: BH over the 3 control-arm helix tests
spec_ps=[spec["caa_vs_baseline"]["mannwhitney_p"],
         spec["matched_control_vs_baseline"]["mannwhitney_p"],
         spec["sham_vs_baseline"]["mannwhitney_p"]]
spec["bh_fdr_helix_tests"]={"labels":["caa_vs_base","mc_vs_base","sham_vs_base"],
                            "p":spec_ps,"q":bh_fdr(spec_ps)}

# ================= C2 non-inferiority (validity) =================
val_base=mean(base,"valid"); val_caa=mean(caa,"valid")
lo_v,hi_v,lb_one=boot_ci_diff(caa["valid"],base["valid"])  # caa - base
MARGIN=0.05
ni_pass = lb_one > -MARGIN
c2={"validity_baseline":val_base,"validity_caa_win":val_caa,"delta":val_caa-val_base,
    "margin":-MARGIN,"ci95_two_sided":[lo_v,hi_v],"one_sided_lower_bound_95":lb_one,
    "non_inferior":bool(ni_pass)}

# off-target winning vs baseline
def offt(k, higher_is_worse=None):
    a,b=caa[k],base[k]
    return {"caa":float(a.mean()),"baseline":float(b.mean()),
            "delta":float(a.mean()-b.mean()),
            "mannwhitney_p":mannwhitney_p(a,b),
            "cliffs_delta":float(cliffs_delta(a,b))}
offtargets={"sheet_frac":offt("sheet"),"gc":offt("gc"),"mean_nll":offt("nll"),
            "prot_len":offt("plen"),"aa_helixfav":offt("aahf")}
off_ps=[offtargets[k]["mannwhitney_p"] for k in offtargets]
off_q=bh_fdr(off_ps)
for k,q in zip(offtargets,off_q): offtargets[k]["q_bh"]=q

# ================= Frontier =================
frontier=[]
for c in sorted(dose28_helix):
    h=dose28_helix[c]; v=dose28_val[c]
    frontier.append({"coef":c,"helix_all":h,"helix_gain_vs_baseline":h-dose28_helix[0.0],
                     "validity_rate":v,"validity_delta_vs_baseline":v-dose28_val[0.0]})

# ================= dump =================
out={
 "C1":{
   "primary_site_family":"site28_caa",
   "trend_test":"permutation on Spearman rho (6 coefs x 3 seeds = 18 pts), one-sided positive",
   "site28":{"rho":rho28,"p_perm":p28,"q_bh":fdr[0]},
   "site30_secondary":{"rho":rho30,"p_perm":p30,"q_bh":fdr[1]},
   "site26_sae_secondary":{"rho":rho26,"p_perm":p26,"q_bh":fdr[2]},
   "winning":{"site":28,"coef":1.0,"helix_all":mean(caa,"helix"),"baseline_helix_all":mean(base,"helix")},
   "effect_size_winning_vs_baseline":{"cliffs_delta":delta_wb,"hedges_g":g_wb,
        "helix_diff_boot_ci95":[lo_wb,hi_wb]},
   "dose_ladder_helix":dose28_helix,"dose_ladder_validity":dose28_val,
 },
 "specificity":spec,
 "C2":{"non_inferiority":c2,"off_targets":offtargets},
 "frontier":frontier,
}
print(json.dumps(out,indent=2))

with open(f"{R}/m4/specificity.json","w") as f: json.dump(spec,f,indent=2)
with open(f"{R}/m4/validity_frontier.json","w") as f:
    json.dump({"site":28,"mode":"caa","margin":MARGIN,
               "rule":"max helix_all s.t. one-sided 95% validity LB > -margin",
               "winning_coef":1.0,"frontier":frontier},f,indent=2)
with open("/data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/experiments/_finalize_out.json","w") as f:
    json.dump(out,f,indent=2)
print("WROTE specificity.json, validity_frontier.json")
