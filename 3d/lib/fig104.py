import json, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B="lc_s03.5_n100_L96_s"; E="_tau8_D1_g0.02_iso"; r103=json.load(open("sum103.json"))
fig,ax=plt.subplots(1,2,figsize=(13,4.8)); col={1:"#2a6fb0",2:"#c0392b"}
for s in (1,2):
    d=r103[f"{B}{s}{E}_pf1"]; ax[0].plot(d["t"],d["ncl"],":",color=col[s],label=f"Матриця {s}: без калібрувального поля (103)")
    g=json.load(open(f"{B}{s}{E}_pf1_gz0.02_g104.json"))["track"]; ax[0].plot([q["t"] for q in g],[q["ncl"] for q in g],"-",color=col[s],lw=2,label=f"Матриця {s}: з калібрувальним полем (104)")
ax[0].set_xlabel("час"); ax[0].set_ylabel("Z₂-вихорів у світах"); ax[0].set_title("Вихори, що народились зі світом"); ax[0].legend(fontsize=7)
for s_ in (1,2):
    for fn,lab,ls in ((f"{B}{s_}{E}_pf1_gz0.02_g104.json","мілкий світ (β=1,5, глибина 6–7)","-"),(f"{B}{s_}{E}_b1.2_pf1_gz0.02_gv100_g104.json","глибокий світ (β=1,2, глибина 16)","--")):
        tr=json.load(open(fn))["track"]; ax[1].plot([q["t"] for q in tr],[q["ncl"] for q in tr],ls,color=col[s_],lw=1.8,label=f"Матриця {s_}: {lab}")
ax[1].set_xlabel("час"); ax[1].set_ylabel("Z₂-вихорів у світі"); ax[1].set_title("Речовина живе лише в глибокому світі (з калібрувальним полем)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig18_vortex_live.png",dpi=130)
