import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(13,4.4))
runs=[("r3d_L16_B_s1_n100_ell3_g0.02","Б, плавне τ (106)","#7f8c8d"),("r3d_L16_B_s1_n100_ell3_g0.02_v2","Б, 2 пари ліній","#2a6fb0"),("r3d_L16_B_s1_n100_ell3_g0.02_v4","Б, 4 пари ліній","#c0392b"),
      ("r3d_L16_B_s1_n100_ell3_g0.02_e1.2_v2","Б, 2 пари, ε₀ = 1,2","#27ae60"),("r3d_L16_A_s1_n100_ell3_g0.02_v2","А, 2 пари ліній","#8e44ad")]
for nm,lab,c in runs:
    w=json.load(open(nm+".json"))["w2"]; w=[r for r in w if "boil" in r and r["t"]<=100]
    ax[0].plot([r["t"] for r in w],[r["boil"]/r["A2"] for r in w],color=c,lw=1,label=lab+f" (світів: {max(r['nB'] for r in w)})")
ax[0].set_xlabel("час"); ax[0].set_ylabel("кипіння (відносне)"); ax[0].set_title("Лінії-вихори до-часу: світів 0 у всіх прогонах"); ax[0].legend(fontsize=8)
D=json.load(open("diag109.json")); a="r3d_L16_B_s1_n100_ell3_g0.02"
lab=["2D (106)","3D, плавне τ","3D, біля ліній\n(2 пари)","3D, далеко\n(2 пари)","3D, біля ліній\n(4 пари)","3D, далеко\n(4 пари)"]
val=[0.31,D[a]["cheap"],D[a+"_v2"]["near"]["cheap"],D[a+"_v2"]["far"]["cheap"],D[a+"_v4"]["near"]["cheap"],D[a+"_v4"]["far"]["cheap"]]
ax[1].bar(range(6),val,color=["gray","#7f8c8d","#2a6fb0","#a9c6e8","#c0392b","#e6a49b"]); ax[1].set_xticks(range(6)); ax[1].set_xticklabels(lab,fontsize=8)
ax[1].set_ylabel("частка «дешевих» вузлів (ціна > −1)"); ax[1].set_title("Біля ліній дешевих місць стільки ж, як у 2D, але світ не народжується")
plt.tight_layout(); plt.savefig("fig23_3d_tau_lines.png",dpi=130)
