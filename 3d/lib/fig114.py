import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=lambda n: json.load(open(n+".json"))["w2"]
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
lab={"all":"FCC (усі трикутники, 24/вузол)","planar":"площини (18/вузол)","layer":"шари (12/вузол)"}
for nm,k,c in [("rgen_all_L16_s1_n100","all","C3"),("rgen_planar_L16_s1_n100","planar","C1"),("rgen_layer_L16_s1_n100","layer","C0"),("rgen_layer_L16_s2_n100","layer","C0")]:
    w=R(nm); ax[0].plot([r["t"] for r in w],[100*r["flat"] for r in w],c,lw=1.5,label=lab[k] if "s2" not in nm else None,ls="--" if "s2" in nm else "-")
ax[0].set_xlabel("час"); ax[0].set_ylabel("пласких вузлів, %"); ax[0].set_title("Спонтанно: впорядковані латки (світів 0)"); ax[0].legend(fontsize=8)
for nm,k,c,ls in [("rgen_layer_L16_s1_n100_pr6","шари, R=6 (925 вузлів)","C0","-"),("rgen_layer_L16_s1_n100_pr7","шари, R=7 (1419)","C0","--"),("rgen_planar_L16_s1_n100_pr7","площини, R=7","C1","-"),("rgen_all_L16_s1_n100_pr7","FCC, R=7","C3","-")]:
    w=R(nm); ax[1].plot([0]+[r["t"] for r in w],[None]+[r["nB"] for r in w],c,ls=ls,marker="o",ms=3,label=k)
ax[1].set_xlim(0,8); ax[1].set_xlabel("час"); ax[1].set_ylabel("вузлів світу"); ax[1].set_title("Посаджений світ гасне за 3 од. часу"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("/mnt/project-files/figures/fig28_3d_lattices.png",dpi=130)
