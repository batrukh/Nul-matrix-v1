import json, re, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
n2={"4.5":69,"6":131}
def ser(fn,key):
    s=open(fn).read(); t=[float(x) for x in re.findall(r"'t': ([0-9.]+)",s)]; v=[int(x) for x in re.findall(rf"'{key}': ([0-9]+)",s)]; return t[:len(v)],v
for R,c in (("2","#7f8c8d"),("3","#2a6fb0"),("4.5","#27ae60"),("6","#c0392b")):
    t,v=ser(f"plant2d_{R}.log","n"); ax[0].plot(t,v,"-",color=c,label=f"2D, радіус {R.replace('.',',')}")
    t,v=ser(f"run3d_pr{R}.log","nB"); ax[0].plot([0]+t,[None]+v,"--",color=c,label=f"3D, радіус {R.replace('.',',')}")
ax[0].set_xlim(0,16); ax[0].set_xlabel("час"); ax[0].set_ylabel("вузлів зі ψ > 0 біля посадки"); ax[0].set_title("Посаджений світ (ψ): у 2D від радіуса 4,5 живе, у 3D гасне за 3 од."); ax[0].legend(fontsize=7,ncol=2)
cols=["#2c3e50","#8e44ad","#2a6fb0","#27ae60","#e67e22","#c0392b"]
for a,c in zip(("0","0.05","0.1","0.15","0.2","0.3"),cols):
    w=json.load(open(f"r3d_L16_B_s1_n100_ell3_g0.02_pr6_pz2_ac{a}.json"))["w2"]
    ax[1].plot([0]+[r["t"] for r in w],[1289/4096]+[r["nB"]/4096 for r in w],color=c,label=f"α = {a.replace('.',',')}")
ax[1].axhline(1,color="k",lw=0.5,ls=":"); ax[1].set_xlabel("час"); ax[1].set_ylabel("частка 3D-Матриці у світі")
ax[1].set_title("3D, повний світ радіуса 6: α ≤ 0,05 з'їдає Матрицю, α ≥ 0,1 гасне"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig24_3d_planted_world.png",dpi=130)
