import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=json.load(open("an105.json")); col={1:"#2a6fb0",2:"#c0392b"}
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
for s in (1,2):
    bs=[1.2,1.3,1.4,1.5]; ax[0].plot(bs,[d[f"{b}_{s}"]["n"] for b in bs],"o-",color=col[s],label=f"Матриця {s}")
    for b,ls in ((1.3,"-"),(1.4,"--")):
        r=d[f"{b}_{s}"]; ax[1].plot(r["t"],r["ncl"],ls,color=col[s],label=f"Матриця {s}, β={str(b).replace('.',',')}")
ax[0].set_xlabel("β"); ax[0].set_ylabel("вихорів у світах (середнє при t ≥ 150)"); ax[0].set_xticks([1.2,1.3,1.4,1.5]); ax[0].set_xticklabels(["1,2","1,3","1,4","1,5"])
ax[0].set_title("Речовина зникає різко між β = 1,4 і 1,5"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("час"); ax[1].set_ylabel("Z₂-вихорів у світах"); ax[1].set_title("Газ вихорів постійно оновлюється (β = 1,3 і 1,4)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig19_vortex_beta.png",dpi=130)
