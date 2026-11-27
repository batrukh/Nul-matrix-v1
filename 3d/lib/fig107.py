import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(13,4.4)); cols=["#7f8c8d","#2a6fb0","#27ae60","#c0392b"]
for e,c in zip((1.2,0.6,0.3,0.15),cols):
    w=json.load(open(f"r3d_L16_B_s1_n100_ell3_g0.02_e{e:g}.json"))["w2"]; lab=f"ε₀ = {e:g} (діюче {e/2:g})"
    ax[0].plot([r["t"] for r in w],[r["nB"] for r in w],color=c,label=lab,lw=1.4)
    ax[1].plot([r["t"] for r in w if "boil" in r],[r["boil"]/r["A2"] for r in w if "boil" in r],color=c,label=lab,lw=1)
ax[0].set_xlabel("час"); ax[0].set_ylabel("вузлів зі ψ > 0 (з 4096)"); ax[0].set_title("Світи з'являються лише при ε₀ = 0,15 і гаснуть за 5–16 од.")
ax[1].set_xlabel("час"); ax[1].set_ylabel("кипіння (відносне)"); ax[1].set_title("Кипіння Матриці від ε₀ не залежить (≈0,26)")
for a in ax: a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig21_3d_eps_sweep.png",dpi=130)
