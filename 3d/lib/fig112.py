import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a=json.load(open("bist112.json")); b=json.load(open("bist112b.json"))
fig,ax=plt.subplots(1,3,figsize=(17,4.6))
H=np.array(a["hist"]["H"]); Hb=a["hist"]["Hb"]; Ab=a["hist"]["Ab"]
ax[0].imshow(np.log10(H.T+1),origin="lower",aspect="auto",extent=[Hb[0],Hb[-1],Ab[0],Ab[-1]],cmap="viridis")
ax[0].set_xlabel("закрученість ниток навколо вузла |голономія|"); ax[0].set_ylabel("інтенсивність |z|²"); ax[0].set_ylim(0,9); ax[0].set_title("Дві хмари: пласкі й інтенсивні проти киплячих")
ax[0].text(0.35,7.6,"← порядок",color="w"); ax[0].text(1.6,1.0,"кипіння",color="w")
h=np.array(b["h"])
for lag,c in (("1","#2a6fb0"),("5","#c0392b")):
    m=np.array([np.nan if x is None else x for x in b["med"][lag]]); ax[1].plot(h,m,"o-",ms=3,color=c,label=f"за {lag} од. часу")
ax[1].axhline(0,color="k",lw=0.8); ax[1].axvline(0.27,color="gray",ls=":"); ax[1].axvline(1.53,color="gray",ls=":")
ax[1].set_xlabel("закрученість зараз"); ax[1].set_ylabel("медіанна зміна закрученості"); ax[1].set_title("Два стани: порядок ≈ 0,27 і кипіння ≈ 1,53"); ax[1].legend(fontsize=8)
lags=[1,5,10]; ax[2].bar([0,1,2],[b["stay"][str(k)] for k in lags],color="#27ae60",label="плаский лишається пласким")
ax[2].bar([4,5,6],[b["enter"][str(k)] for k in lags],color="#c0392b",label="киплячий стає пласким")
ax[2].axhline(b["base"],color="k",ls="--",lw=0.8,label=f"частка пласких у Матриці ({b['base']*100:.0f} %)")
ax[2].set_xticks([0,1,2,4,5,6]); ax[2].set_xticklabels(["1","5","10","1","5","10"]); ax[2].set_xlabel("через скільки одиниць часу"); ax[2].set_ylabel("імовірність")
ax[2].set_title("Порядок тримається, перехід у нього рідкісний"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig26_bistability.png",dpi=130)
