import json, os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import ana60 as A
W = lambda nm: [r for r in json.load(open(nm + ".json"))["w2"] if "flat" in r]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
runs = [("rgen_tet_L16_s1_n100", "А, α = 1 (3 реалізації)", "0.55", "-"), ("rgen_tet_L16_s2_n100", None, "0.55", "-"), ("rgen_tet_L16_s3_n100", None, "0.55", "-"),
        ("rgen_tet_L16_s1_n100_ac0.45", "А, α = 0,45", "C2", "-"), ("rgen_tet_L16_s1_n100_ac0.4", "А, α = 0,4", "C1", "-"), ("rgen_tet_L16_s1_n100_ac0.3", "А, α = 0,3", "C4", "-"),
        ("rq_L16_s1_n100", "Б (тетраедри), α = 1", "C3", "-"), ("rq_L16_s2_n100", None, "C3", "-"), ("rq_L16_s3_n100", None, "C3", "-"),
        ("rq_L16_s1_n100_ac0.6", "Б, α = 0,6", "C9", "-"), ("rgen_layer_L16_s1_n100", "«шари», α = 1 (114)", "k", ":")]
runs = [r for r in runs if os.path.exists(r[0] + ".json")]
for nm, lab, c, ls in runs:
    w = W(nm); ax[0].plot([r["t"] for r in w], [100*r["flat"] for r in w], color=c, ls=ls, lw=1.5, label=lab)
ax[0].axhspan(9, 11, color="C0", alpha=0.15); ax[0].text(60, 12, "2D: 9–11 %", fontsize=8, color="C0")
ax[0].set_xlabel("час"); ax[0].set_ylabel("пласкі вузли, %"); ax[0].set_yscale("symlog", linthresh=1); ax[0].set_title("Порядок у 3D: світів 0 у всіх прогонах"); ax[0].legend(fontsize=7, ncol=2)
for nm, lab, c, ls in runs:
    if "layer" in nm or not os.path.exists(nm + ".npz"): continue
    o = A.flat_clusters(nm, "tet"); ax[1].scatter(o["sizes"][0], o["psi_big"][0], color=c, s=40, edgecolor="k", zorder=3)
ax[1].axvspan(95, 372, color="C0", alpha=0.15); ax[1].axhline(-0.6, color="C0", ls="--", lw=1)
ax[1].text(100, -0.62, "2D: народження з ділянок 95–372 вузли при ψ ≈ −0,6", fontsize=8, color="C0", va="top")
ax[1].set_xscale("log"); ax[1].set_xlabel("найбільша пласка ділянка, вузлів (t = 100)"); ax[1].set_ylabel("ψ у ній"); ax[1].set_ylim(-0.95, -0.5)
ax[1].set_title("Розмір протосвітів є, підйому ψ немає")
for fn, lab, c in [("bal_plant2d_pr6", "2D, R = 6 (живе)", "C0"), ("bal_rgen_layer_L16_s1_n100_pr6", "«шари», R = 6", "k"),
                   ("bal_rq_L16_s1_n100_pr3.5", "Б, R = 3,5 (179 вузлів)", "C3"), ("bal_rq_L16_s2_n100_pr4.5", "Б, R = 4,5 (389 вузлів)", "C1")]:
    d = json.load(open(fn + ".json")); b = d["bal"] if isinstance(d, dict) else d; b = [r for r in b if r["t"] <= 2.0]
    ax[2].plot([r["t"] for r in b], [r["psi"] for r in b], color=c, lw=1.8, label=lab + ": ψ")
    ax[2].plot([r["t"] for r in b], [r["cost"] for r in b], color=c, lw=1.2, ls="--")
ax[2].axhline(0, color="0.6", lw=0.8); ax[2].set_ylim(-3, 1.2); ax[2].set_xlabel("час після посадки"); ax[2].set_ylabel("ψ (суцільна), ціна ε₀ (штрих)")
ax[2].set_title("Посаджений світ при α = 1: ціна ε₀ гасить ψ"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig("/mnt/project-files/figures/fig31_tetra_matrix.png", dpi=130)
