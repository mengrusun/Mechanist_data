import sys, json, os
from collections import defaultdict
sys.path.insert(0, os.path.abspath("figures"))
from _style import plt, C, save
OUT="figures/C1"
series=[("results/m3/s28_caa.json","Site 28 · CAA (primary)",C[0],"o","-"),
        ("results/m3/s30_caa.json","Site 30 · CAA",C[1],"s","--"),
        ("results/m3/s26_sae_clamp.json","Site 26 · SAE-clamp",C[3],"^",":")]
fig,ax=plt.subplots(figsize=(5,3.4))
base=None
for path,lab,col,mk,ls in series:
    d=json.load(open(path)); rows=d["rows"]
    agg=defaultdict(list)
    for r in rows: agg[r["coef"]].append(r["helix_all"])
    xs=sorted(agg); ys=[sum(agg[c])/len(agg[c]) for c in xs]
    if base is None: base=ys[0]  # coef 0 helix = baseline
    ax.plot(xs,ys,marker=mk,ls=ls,color=col,label=lab,lw=1.6,ms=5)
ax.axhline(base,color="0.5",lw=1,ls=(0,(2,2)))
ax.text(8,base+0.006,f"baseline {base:.3f}",color="0.4",ha="right",va="bottom",fontsize=8)
ax.axvline(1.0,color="0.75",lw=1,ls=(0,(1,2)))
ax.text(1.0,0.965,"winning coef 1.0",color="0.45",ha="center",va="top",fontsize=7,rotation=90,transform=ax.get_xaxis_transform())
ax.set_xlabel("Steering coefficient")
ax.set_ylabel(r"$\alpha$-helix fraction (all gens, invalid$\to$0)")
ax.set_xticks([0,0.5,1,2,4,8]); ax.legend(frameon=False,loc="lower left",fontsize=8)
save(fig,"c1_dose_response",OUT)
