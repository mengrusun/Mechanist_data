import sys, json, os
sys.path.insert(0, os.path.abspath("figures"))
from _style import plt, C, save
OUT="figures/C2"
d=json.load(open("results/m4/validity_frontier.json")); fr=d["frontier"]; margin=d.get("margin",0.05)
xs=[r["coef"] for r in fr]; gain=[r["helix_gain_vs_baseline"] for r in fr]; val=[r["validity_rate"] for r in fr]
base_val=val[0]
fig,ax=plt.subplots(figsize=(5,3.4))
l1,=ax.plot(xs,gain,marker="o",color=C[0],lw=1.7,ms=5,label=r"$\alpha$-helix gain vs baseline")
ax.set_xlabel("Steering coefficient"); ax.set_ylabel(r"$\alpha$-helix gain",color=C[0])
ax.tick_params(axis="y",labelcolor=C[0]); ax.set_xticks(xs)
ax.axhline(0,color="0.7",lw=0.8)
ax2=ax.twinx(); ax2.spines["top"].set_visible(False)
l2,=ax2.plot(xs,val,marker="s",color=C[3],lw=1.7,ms=5,ls="--",label="Validity rate")
ax2.set_ylabel("Validity rate",color=C[3]); ax2.tick_params(axis="y",labelcolor=C[3])
ni=base_val-margin
ax2.axhline(ni,color=C[3],lw=1,ls=(0,(1,2)))
ax2.text(8,ni,f" NI margin {ni:.3f}",color=C[3],ha="right",va="bottom",fontsize=7)
ax.axvline(d.get("winning_coef",1.0),color="0.75",lw=1,ls=(0,(1,2)))
ax.text(d.get("winning_coef",1.0),0.02,"winning\ncoef 1.0",color="0.45",ha="center",va="bottom",fontsize=7,transform=ax.get_xaxis_transform())
ax.legend(handles=[l1,l2],frameon=False,loc="upper left",fontsize=8)
save(fig,"c2_validity_frontier",OUT)
