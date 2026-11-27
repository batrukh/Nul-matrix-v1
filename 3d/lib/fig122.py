import json, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B, O, INK, MUT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUT, "xtick.color": MUT, "ytick.color": MUT, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4)); c = lambda x: f"{x:g}".replace(".", ",")
b = json.load(open("ana122b.json")); ks = ["1.3_1", "1.3_2", "1.5_1", "1.5_2"]; x = np.arange(len(ks))
ax[0].bar(x - 0.2, [100*b[k]["b_wall"] for k in ks], 0.38, color=O, label="місця народжень"); ax[0].bar(x + 0.2, [100*b[k]["r_wall"] for k in ks], 0.38, color=B, label="випадкові місця світу")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"β {c(float(k[:3]))}, М{k[-1]}" for k in ks]); ax[0].set_ylabel("% ближче 3 до стінки Φ★ | −Φ★"); ax[0].legend(fontsize=7, frameon=True, loc="center right")
ax[0].set_title("а) вихори народжуються на стінці між вакуумами", fontsize=9)
e = json.load(open("ana122c.json")); bs = sorted({float(k.split("_")[0]) for k in e})
for s, mk in ((1, "o"), (2, "s")):
    kk = [f"{c(bb).replace(',', '.')}_{s}" for bb in bs]; kk = [k for k in kk if k in e]; xb = [float(k.split("_")[0]) for k in kk]
    ax[1].plot(xb, [e[k][0] for k in kk], mk, color=O, ms=8, mec="white", label=f"усі, Матриця {s}"); ax[1].plot(xb, [e[k][1] for k in kk], mk, color=B, ms=8, mec="white", label=f"поза стінкою, Матриця {s}")
ax[1].set_xticks(bs); ax[1].set_xticklabels([c(v) for v in bs], fontsize=7); ax[1].set_xlabel("β"); ax[1].set_ylabel("вихорів у кінцевому стані"); ax[1].legend(fontsize=7, frameon=False)
ax[1].set_title("б) поза стінкою вихорів 0–3 при будь-якому β", fontsize=9)
for f, ls in zip(sorted(glob.glob("*b1.3*_frz120_p0_aw.json")), ("-", "--")):
    tr = json.load(open(f))["track"]; s = f.split("_L96_s")[1][0]; t = [r["t"] for r in tr]
    ax[2].plot(t, [r["ncl"] for r in tr], ls, color=O, lw=2, label=f"усі, Матриця {s}"); ax[2].plot(t, [r["npure"] for r in tr], ls, color=B, lw=2, label=f"поза стінкою, Матриця {s}")
ax[2].set_xlabel("час після заморожування ψ"); ax[2].set_ylabel("вихорів"); ax[2].legend(fontsize=7, frameon=False); ax[2].set_title("в) β = 1,3, ψ заморожено: «газ» тримається лише на стінці", fontsize=9)
plt.tight_layout(); plt.savefig("fig36_vacuum_walls.png", dpi=130)
