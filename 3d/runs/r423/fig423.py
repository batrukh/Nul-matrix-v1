import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = json.load(open("ana423.json"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
for d in D:
    if d["a"] == 0: continue
    c = "#2a7" if d["p"] >= 0.75 else "#c33"; mk = "s" if d["s0"] != 3.0 else "o"
    lab = (f"α {d['a']:g}" + ("" if d["s0"] == 3.0 else f", s0 {d['s0']:g}")).replace(".", ",")
    ax[0].errorbar(d["T3"], d["p"], xerr=2*d["T3e"], fmt=mk, color=c, ms=8, capsize=3)
    ax[0].annotate(lab, (d["T3"], d["p"]), textcoords="offset points", xytext=(6, 8) if d["p"] > 0.5 else ((-60, -16) if d["s0"] != 3.0 else ((6, 8) if d["a"] == 0.25 else (6, -16))), fontsize=8)
    ax[1].plot(d["psim"], d["p"], mk, color=c, ms=8); ax[1].annotate(lab, (d["psim"], d["p"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax[0].set_xscale("log"); ax[0].set_xlabel("ефективна температура кипіння T_eff = Δ/χ (τ = 3)"); ax[0].set_ylabel("частка зерен, де народився світ")
ax[0].set_title("(423) Народження НЕ визначає T_eff\n(зелене — світ є, червоне — нема; смуги ±2 SEM)", fontsize=10)
ax[1].axvline(-0.848, ls="--", color="gray"); ax[1].set_xlabel("середнє ψ Матриці на t = 6"); ax[1].set_title("Підказка після перегляду (гіпотеза):\nглибина Матриці (середнє ψ) розділяє точки", fontsize=10)
ax[1].text(0.35, 0.45, "α 0 (без кипіння): −0,786, 0/8\n— не показано", transform=ax[1].transAxes, fontsize=8)
for a in ax: a.set_ylim(-0.1, 1.15); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig293.png", dpi=150); import shutil; shutil.copy("fig293.png", "/mnt/project-files/figures/fig293_temperatura_kypinnia.png")
