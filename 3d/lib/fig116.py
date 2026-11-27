import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11,4.2)); A=["0.3","0.4","0.5","0.6","0.7","0.85"]; cm=plt.cm.viridis
fin=[];boil=[]
for i,a in enumerate(A):
    w=[r for r in json.load(open(f"rgen_layer_L16_s1_n100_ac{a}.json"))["w2"] if "nB" in r]
    ax[0].plot([r["t"] for r in w],[100*r["nB"]/4096 for r in w],color=cm(i/5),lw=1.8,label=f"α = {a.replace('.',',')}")
    fin.append(100*sum(r["nB"] for r in w[-20:])/20/4096); boil.append(sum(r["boil"] for r in w[-20:])/20)
for nm,lab,c,ls in [("rgen_layer_L16_s1_n100_ac0.45","α = 0,45 (2 світи)","C1","-"),("rgen_layer_L16_s2_n100_ac0.4","α = 0,4, реалізації 2 і 3","C3",":"),("rgen_layer_L16_s3_n100_ac0.4",None,"C3",":")]:
    w=[r for r in json.load(open(nm+".json"))["w2"] if "nB" in r]; ax[0].plot([r["t"] for r in w],[100*r["nB"]/4096 for r in w],color=c,ls=ls,lw=1.8,label=lab)
w=[r for r in json.load(open("rgen_layer_L16_s1_n100.json"))["w2"] if "nB" in r]
fin.append(0.0); boil.append(sum(r.get("boil",0) for r in w[-20:])/20); A2=[float(a) for a in A]+[1.0]
w=[r for r in json.load(open("rgen_layer_L16_s1_n100_ac0.45.json"))["w2"] if "nB" in r]; fin.insert(2,100*sum(r["nB"] for r in w[-20:])/20/4096); boil.insert(2,sum(r["boil"] for r in w[-20:])/20); A2.insert(2,0.45)
ax[0].set_xlabel("час"); ax[0].set_ylabel("частка коробки у світах, %"); ax[0].set_title("3D «шари», без посадки (реалізація 1, якщо не сказано)"); ax[0].legend(fontsize=8)
ax[1].plot(A2,fin,"o-",c="C0",label="частка у світах, % (t = 80–100)"); ax[1].axhline(4,c="k",ls=":",lw=1); ax[1].text(0.75,5.5,"2D ≈ 4 %",fontsize=8)
ax2=ax[1].twinx(); ax2.plot(A2,boil,"s--",c="C3",label="кипіння"); ax2.set_ylabel("кипіння",color="C3")
ax[1].set_xlabel("поворот α"); ax[1].set_ylabel("частка у світах, %",color="C0"); ax[1].set_title("Реалізація 1: вікно скінченних світів α ≈ 0,4–0,45")
plt.tight_layout(); plt.savefig("/mnt/project-files/figures/fig30_3d_alpha_scan.png",dpi=130)
