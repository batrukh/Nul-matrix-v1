import os; os.environ["NB"]="0"
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import local60 as LC, rnd51 as R, diag3d as D3
fig,ax=plt.subplots(1,2,figsize=(13,4.6)); col={"A":"#2a6fb0","B":"#c0392b"}
for v in ("A","B"):
    for sd,ls in ((1,"-"),(2,"--")):
        w=json.load(open(f"r3d_L16_{v}_s{sd}_n100_ell3_g0.02.json"))["w2"]
        ax[0].plot([r["t"] for r in w if "boil" in r],[r["boil"]/r["A2"] for r in w if "boil" in r],ls,color=col[v],label=f"3D, варіант {'А' if v=='A' else 'Б'}, реалізація {sd}")
w2=json.load(open("lc_s03.5_n100_L96_s1_tau8_D1_g0.02_iso.json"))["w2"]
ax[0].plot([r["t"] for r in w2 if "boil" in r],[r["boil"]/r["A2"] for r in w2 if "boil" in r],":",color="gray",label="2D L96 (для порівняння)")
ax[0].set_xlabel("час"); ax[0].set_ylabel("кипіння (відносна зміна |z|² за одиницю часу)"); ax[0].set_title("3D-Матриця кипить, світів 0 (усі 4 прогони)"); ax[0].legend(fontsize=7)
d=np.load("lc_s03.5_n100_L96_s1_tau8_D1_g0.02_iso.npz"); s=LC.Loc(L=96,h=0.3,lc=1.3); f=s.fab; R.flip(f,d["bad"]); s.m0=3.5
c2=s.F(d["z"],np.zeros(f.n),parts=True)["cost"][d["psi"]<0]
d3=np.load("r3d_L16_B_s1_n100_ell3_g0.02.npz"); s3=D3.build3(16,"B",1); R.flip(s3.fab,d3["bad"]); s3.m0=3.5
c3=s3.F(d3["z"],np.zeros(s3.fab.n),parts=True)["cost"]
bins=np.linspace(-60,2,63)
ax[1].hist(c2,bins=bins,density=True,alpha=0.6,color="gray",label="2D: вузол у 6 трикутниках")
ax[1].hist(c3,bins=bins,density=True,alpha=0.6,color="#c0392b",label="3D (Б): вузол у 24 трикутниках")
ax[1].axvline(-0.3,color="k",lw=0.8); ax[1].set_yscale("log")
ax[1].set_xlabel("ціна входу у світ для вузла (сила на ψ при ψ = 0)"); ax[1].set_ylabel("густина"); ax[1].set_title("У 3D зникають «дешеві місця» для народження"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig20_3d_matrix.png",dpi=130)
