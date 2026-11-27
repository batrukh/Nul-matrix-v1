import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(16,4.4)); qs=("0","0.3","3","10"); cols=["#2c3e50","#2a6fb0","#27ae60","#c0392b"]
lab={"0":"0 (правильна сітка)","0.3":"0,3 (трохи безладу)","3":"3 (випадкова)","10":"10 (повністю випадкова)"}
for q,c in zip(qs,cols):
    w=json.load(open(f"rfl_L96_q{q}_g0_s1_n100_ell8.json"))["w2"]
    ax[0].plot([r["t"] for r in w],[r["nB"]/9216 for r in w],color=c,label="перекидань на нитку "+lab[q])
    ax[1].plot([r["t"] for r in w if "boil" in r],[r["boil"]/r["A2"] for r in w if "boil" in r],color=c,lw=1)
ax[0].set_xlabel("час"); ax[0].set_ylabel("частка вузлів у світах (ψ > 0)"); ax[0].set_title("Світи народжуються лише на впорядкованій Матриці"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("час"); ax[1].set_ylabel("кипіння (відносне)"); ax[1].set_title("Безлад посилює кипіння")
D=json.load(open("diagflip.json"))
for q,c in zip(("0.3","3","10"),cols[1:]):
    cst=np.array(D[q]["cost"]); dg=np.array(D[q]["deg"]); ks=[k for k in range(3,15) if (dg==k).sum()>20]
    ax[2].plot(ks,[(cst[dg==k]>-1).mean() for k in ks],"o-",color=c,label="перекидань "+q.replace(".",","))
c0=np.array(D["0"]["cost"]); ax[2].axhline((c0>-1).mean(),color=cols[0],ls=":",label="правильна сітка (кожен вузол у 6)")
ax[2].set_xlabel("трикутників на вузол"); ax[2].set_ylabel("частка «дешевих» вузлів Матриці (ціна > −1)"); ax[2].set_title("«Тонкі» вузли дешевші, але Матриця в цілому дорожчає"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig22_random_matrix.png",dpi=130)
