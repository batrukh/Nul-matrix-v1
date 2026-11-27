import json, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = json.load(open("tab120.json"))      # (β, стан, N нов. 30–150, N нов. ≥150, N стар. ≥150, f3, wtri, max)
B, O, INK, MUT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUT, "axes.labelcolor": INK, "xtick.color": MUT, "ytick.color": MUT, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
c = lambda x: f"{x:g}".replace(".", ",")
for s, mk in ((1, "o"), (2, "s")):
    r = [q for q in rows if q[1] == s]; b = [q[0] for q in r]
    ax[0].plot(b, [q[4] for q in r], mk, color=O, ms=8, mec="white", label=f"старий детектор, Матриця {s}")
    ax[0].plot(b, [q[3] for q in r], mk + "-", color=B, ms=8, mec="white", lw=2, label=f"вакуум-свідомий, Матриця {s}")
ax[0].set_xlabel("β"); ax[0].set_ylabel("Z₂-вихорів у світах (середнє, t ≥ 150)"); bs = sorted({q[0] for q in rows}); ax[0].set_xticks(bs); ax[0].set_xticklabels([c(x) for x in bs], fontsize=7)
ax[0].set_title("а) справжні вихори є при β ≤ 1,3; «газ» при 1,4–1,48 — хибний", fontsize=9); ax[0].legend(fontsize=7, frameon=False)
ax[1].plot([q[5] for q in rows], [q[4] for q in rows], "o", color=O, ms=8, mec="white", label="старий детектор")
ax[1].plot([q[5] for q in rows], [q[3] for q in rows], "o", color=B, ms=8, mec="white", label="вакуум-свідомий")
ax[1].set_xlabel("частка світу у вакуумі −Φ★ (три додатні власні значення)"); ax[1].set_ylabel("вихорів (t ≥ 150)")
ax[1].set_title("б) старий лічильник іде за часткою другого вакууму, β 1,2–1,5", fontsize=9); ax[1].legend(fontsize=7, frameon=False)
f = glob.glob("*_s2_*_b1.44_*_gk1_gd_g104.json")[0]; tr = json.load(open(f))["track"]
t = [q["t"] for q in tr]
ax[2].plot(t, [q["ncl_old"] for q in tr], color=O, lw=2, label="старий детектор"); ax[2].plot(t, [q["ncl"] for q in tr], color=B, lw=2, label="вакуум-свідомий")
ax[2].set_xlabel("час"); ax[2].set_ylabel("Z₂-вихорів у світах"); ax[2].set_title("в) β = 1,44, Матриця 2: ранні вихори Кіббла справжні, потім хибні", fontsize=9); ax[2].legend(fontsize=7, frameon=False, loc="center left")
plt.tight_layout(); plt.savefig("fig34_vacuum_detector.png", dpi=130)
