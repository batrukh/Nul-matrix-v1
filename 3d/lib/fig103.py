import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, phi90 as PH
r=json.load(open("sum103.json")); B="lc_s03.5_n100_L96_s"; E="_tau8_D1_g0.02_iso_pf1"
fig,ax=plt.subplots(1,3,figsize=(16,4.8))
sty={"":("-",None),"p1":("--",None),"e0.01":("-",0.55),"e0.1":("-",0.25)}
col={1:"#2a6fb0",2:"#c0392b"}
for s in (1,2):
    for suf,lab in (("","без спадковості"),("p1","інший шум Φ★"),("e0.01","спадковість ε=0,01"),("e0.1","спадковість ε=0,1")):
        d=r[f"{B}{s}{E}{suf}"]; ls,al=sty[suf]
        ax[0].plot(d["t"],d["ncl"],ls,color=col[s],alpha=al or 1,lw=1.6 if not al else 1.2,label=f"Матриця {s}: {lab}")
ax[0].set_xlabel("час"); ax[0].set_ylabel("Z₂-вихорів у світах"); ax[0].legend(fontsize=7); ax[0].set_title("Скільки вихорів живе у світах")
L=96; X,Y=PH.coords(L)
for k,suf in enumerate(("","p1")):
    z=np.load(f"{B}2{E}{suf}_snap.npz"); i=1; psi=z["psi"][i].reshape(L,L)
    a=ax[1+k]; a.scatter(X.ravel(),Y.ravel(),c=np.where(psi.ravel()>0.3,0.8,0.15),cmap="Greys",vmin=0,vmax=1,s=2,marker="h",linewidths=0)
    cl=r[f"{B}2{E}{suf}"]["cl"][i]; 
    if cl: a.scatter(*np.array(cl).T,c="#c0392b",s=40,marker="x")
    a.set_title(f"Матриця 2, t = 10, {'шум Φ★ №0' if not suf else 'шум Φ★ №1'}: {len(cl)} вихорів"); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig("fig17_kibble.png",dpi=130)
