import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D2 = {'psi': -0.667, 'A2': 6.06, 'hol': 0.248, 'pot': -0.013, 'grad': -0.086, 'cost': -0.605, 'coup': 0.728}   # diagp.py 2d
D3 = {'psi': -0.85, 'A2': 5.28, 'hol': 0.298, 'pot': 0.077, 'grad': -0.165, 'cost': -0.41, 'coup': 0.506}     # diagp.py 3d 1
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4)); k = ["coup", "cost", "grad", "pot"]; lab = ["зв'язок λ_c", "ціна ε₀", "градієнт κ", "потенціал"]; x = np.arange(4)
ax[0].bar(x - 0.2, [D2[q] for q in k], 0.4, label=f"2D, ψ = {D2['psi']:.2f}".replace(".", ","), color="C0"); ax[0].bar(x + 0.2, [D3[q] for q in k], 0.4, label=f"3D Б, ψ = {D3['psi']:.2f}".replace(".", ","), color="C3")
ax[0].axhline(0, c="k", lw=0.8); ax[0].set_xticks(x, lab); ax[0].set_ylabel("сила на ψ (середнє по ділянках)"); ax[0].legend(fontsize=8)
ax[0].set_title("Сили на ψ у природних протосвітах (рівновага)")
ps = np.linspace(-1.1, 0.2, 200); ax[1].plot(ps, -(ps**3 - ps) + 0.3, c="k", label="потенціал −V′(ψ), h = 0,3")
for D, c, n in ((D2, "C0", "2D"), (D3, "C3", "3D Б")):
    ax[1].axvline(D["psi"], c=c, ls="--"); ax[1].text(D["psi"] + 0.01, 0.42, n, color=c, fontsize=9)
ax[1].axvline(-1/np.sqrt(3), c="0.5", ls=":"); ax[1].text(-0.56, 0.2, "вершина бар'єра\nψ ≈ −0,58", fontsize=8, color="0.4")
ax[1].axhline(0, c="k", lw=0.6); ax[1].set_xlabel("ψ"); ax[1].set_ylabel("−V′(ψ)"); ax[1].set_title("2D-протосвіти біля вершини бар'єра, 3D — глибше"); ax[1].legend(fontsize=8)
for fn, c, n in (("diagz_2d_pr6.0", "C0", "2D, посадка R = 6"), ("diagz_3d_seed1_pr3.5", "C3", "3D Б, посадка R = 3,5")):
    d = json.load(open(fn + ".json")); t = [r["t"] for r in d]
    ax[2].plot(t, [r["A2"] for r in d], c=c, lw=2, label=n + ": |z|² ниток")
    ax[2].plot(t, [r["psi"] for r in d], c=c, lw=1.2, ls="--", label=n + ": ψ")
ax[2].axhline(0, c="k", lw=0.6); ax[2].set_xlabel("час після посадки"); ax[2].set_title("Посадка: у 2D нитки слабкі від початку"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig("/mnt/project-files/figures/fig32_psi_forces.png", dpi=130)
