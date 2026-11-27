import json, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B, O, INK, MUT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUT, "xtick.color": MUT, "ytick.color": MUT, "axes.spines.top": False, "axes.spines.right": False})
S = json.load(open("sum121.json")); fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for k, (nm, ttl) in enumerate([("lc_s03.5_n100_L96_s2_tau8_D1_g0.02_iso_pf1", "а) (103), Матриця 2: «кипіння» було хибним"), ("lc_s03.5_n100_L96_s1_tau8_D1_g0.02_iso_pf1", "б) (103), Матриця 1: вихори Кіббла справжні")]):
    r = S[nm]; ax[k].plot(r["t"], r["old"], color=O, lw=2, label="старий детектор"); ax[k].plot(r["t"], r["new"], color=B, lw=2, label="вакуум-свідомий")
    ax[k].set_xlabel("час"); ax[k].set_ylabel("Z₂-вихорів у світах"); ax[k].set_title(ttl, fontsize=9); ax[k].legend(fontsize=7, frameon=False)
for s, ls in ((1, "-"), (2, "--")):
    tr = json.load(open(glob.glob(f"*_s{s}_*_iso_pf1_gz0.02_gv100_gk1_gd_g104.json")[0]))["track"]
    t = [q["t"] for q in tr if q["t"] >= 100]; d = [q["near"] if q["near"] is not None else np.nan for q in tr if q["t"] >= 100]
    ax[2].plot(t, d, ls, color=B, lw=2, label=f"Матриця {s}")
ax[2].axhline(4, color=MUT, lw=1, ls=":"); ax[2].set_ylim(0, 12); ax[2].set_xlabel("час (посадка при t = 100)"); ax[2].set_ylabel("відстань до найближчого вихору від місця посадки")
ax[2].set_title("в) (104), β = 1,5: посаджений вихор живе ≈40 і ≥150 од.", fontsize=9); ax[2].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig("fig35_recount_103_104.png", dpi=130)
