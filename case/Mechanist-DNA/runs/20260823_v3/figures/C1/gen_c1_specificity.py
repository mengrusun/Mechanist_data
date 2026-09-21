import sys, json, os
sys.path.insert(0, os.path.abspath("figures"))
from _style import plt, C, save
OUT="figures/C1"
d=json.load(open("results/m4/specificity.json")); m=d["means"]
order=[("baseline","Baseline",C[7]),("caa_win","CAA\n(winning)",C[0]),
       ("matched_control","Matched\ncontrol",C[2]),("sham","Sham",C[4])]
labs=[o[1] for o in order]; vals=[m[o[0]] for o in order]; cols=[o[2] for o in order]
fig,ax=plt.subplots(figsize=(4.4,3.2))
bars=ax.bar(labs,vals,color=cols,width=0.66,edgecolor="0.2",lw=0.5)
base=m["baseline"]
ax.axhline(base,color="0.5",lw=1,ls=(0,(2,2)))
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2,v+0.004,f"{v:.3f}",ha="center",va="bottom",fontsize=8)
ax.set_ylabel(r"$\alpha$-helix fraction (all gens)")
ax.set_ylim(0,max(vals)*1.16)
ax.text(0.5,base+0.004,"CAA +0.068 vs baseline; controls: no gain",fontsize=7.5,color="0.35",ha="left",va="bottom",transform=ax.get_yaxis_transform() if False else ax.transData)
save(fig,"c1_specificity",OUT)
