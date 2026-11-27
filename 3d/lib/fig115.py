import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
J=lambda n: json.load(open(n+".json"))
fig,ax=plt.subplots(1,3,figsize=(15,4.3))
for nm,lab,c in [("bal_plant2d_pr6","2D, R=6","k"),("bal_rgen_layer_L16_s1_n100_pr6","3D шари, R=6","C0"),("bal_rgen_all_L16_s1_n100_pr7","3D FCC, R=7","C3")]:
    b=J(nm)["bal"]; t=[r["t"] for r in b]
    ax[0].plot(t,[-r["cost"] for r in b],c,lw=1.8,label=lab+": ціна ε₀ (−)")
    ax[0].plot(t,[r["pot"]+r["coup"] for r in b],c,ls=":",lw=1.5,label=lab+": потенціал + λc")
    ax[1].plot(t,[r["hol"] for r in b],c,lw=1.8,label=lab)
ax[0].set_yscale("log"); ax[0].set_xlim(0,3); ax[0].set_xlabel("час після посадки"); ax[0].set_ylabel("сила на ψ, середнє в кулі"); ax[0].set_title("Що штовхає ψ вниз і що тримає"); ax[0].legend(fontsize=7)
ax[1].set_xlim(0,3); ax[1].set_xlabel("час після посадки"); ax[1].set_ylabel("|голономія| трикутників у кулі"); ax[1].set_title("У 2D нитки під світом вирівнюються, у 3D — ні"); ax[1].legend(fontsize=8)
for nm,lab,c,ls in [("rgen_layer_L16_s1_n100_ac0.3","шари, α=0,3, без посадки","C0","-"),("rgen_layer_L16_s1_n100_ac0.3_pr6","шари, α=0,3, посадка R=6","C0","--"),("rgen_all_L16_s1_n100_ac0.3_pr7","FCC, α=0,3, посадка R=7","C3","--")]:
    d=J(nm); w=[r for r in d["w2"] if "nB" in r]; ax[2].plot([r["t"] for r in w],[100*r["nB"]/4096 for r in w],c,ls=ls,lw=1.8,label=lab)
ax[2].set_xlabel("час"); ax[2].set_ylabel("частка коробки у світі, %"); ax[2].set_title("α = 0,3: світ живе (у 2D частка ≈ 4 %)"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("/mnt/project-files/figures/fig29_3d_balance.png",dpi=130)
