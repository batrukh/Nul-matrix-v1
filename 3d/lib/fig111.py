import os; os.environ["NB"]="0"
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import local60 as LC, fab3d as F, rnd51 as R
from diag111 import node_feats
r=json.load(open("birth111.json"))["res"]
names={"hol":"закрученість ниток\n(голономія)","A2":"інтенсивність\nниток","phsync":"узгодженість\nповоротів фаз","psi":"ψ","boil":"кипіння","sync":"такт зміни\nінтенсивності","cost":"ціна входу\nу світ (ψ = 0)","extr":"екстремум τ\n(усі стріли в/з)","A2sd":"розкид\nінтенсивності","holsd":"розкид\nзакрученості"}
keys=["hol","A2","phsync","psi","boil","sync","cost","extr","A2sd","holsd"]
fig,ax=plt.subplots(1,2,figsize=(15,5.4)); x=np.arange(len(keys))
for j,(lag,c) in enumerate((("5","#2a6fb0"),("10","#c0392b"))):
    ax[0].bar(x+(j-0.5)*0.38,[r[lag][k]["auc"]-0.5 for k in keys],0.38,bottom=0.5,color=c,label=f"за {lag} од. до народження")
ax[0].axhline(0.5,color="k",lw=0.8); ax[0].set_xticks(x); ax[0].set_xticklabels([names[k] for k in keys],fontsize=7,rotation=35,ha="right")
ax[0].set_ylabel("AUC (0,5 — ознака нічого не каже)"); ax[0].set_ylim(0,1); ax[0].legend(fontsize=8)
ax[0].set_title("2D: місце народження заздалегідь пласке, інтенсивне й узгоджене")
s=LC.Loc(L=96,h=0.3,lc=1.3); f=s.fab; t0,h0=f.tail.copy(),f.head.copy(); H2=[]
for sd in (1,2,3,4):
    d=np.load(f"lc_s03.5_n100_L96_s{sd}_tau8_D1_g0.02_iso_sn_sn.npz")
    for k in range(10,60,5):
        tl=d["tail"][k]; fl=tl!=t0; f.tail=tl; f.head=np.where(fl,t0,h0); A,H=node_feats(f,d["z"][k].astype(complex)); H2+=list(H[d["psi"][k]<-0.3])
d=np.load("r3d_L16_B_s1_n100_ell3_g0.02.npz"); f3=F.Fabric3D(16); R.flip(f3,d["bad"]); A3,H3=node_feats(f3,d["z"])
bins=np.linspace(0,np.pi,50)
ax[1].hist(H2,bins,density=True,alpha=0.6,color="gray",label="2D-Матриця (кипить, світи народжуються)")
ax[1].hist(H3,bins,density=True,alpha=0.6,color="#c0392b",label="3D-Матриця (кипить, світів немає)")
ax[1].axvline(0.35,color="k",lw=0.8,ls="--"); ax[1].text(0.38,ax[1].get_ylim()[1]*0.9,"пласкі ділянки",fontsize=8)
ax[1].set_xlabel("середня закрученість ниток навколо вузла |голономія|"); ax[1].set_ylabel("густина")
ax[1].set_title("У 2D-Матриці 11 % вузлів пласкі, у 3D — 0,3 %"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig25_birth_trigger.png",dpi=130)
