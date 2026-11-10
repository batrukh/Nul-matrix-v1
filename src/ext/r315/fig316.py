# fig211: (316) що гасить світи у склі — шум стріл чи граф; зерна порядку
import json, glob, sys, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/mnt/project-files/model/s52/r315"; sys.path.insert(0, "/mnt/project-files/model/s52/r268/code"); import rnd51 as R
def load(d):
    D = json.load(open(glob.glob(f"{B}/{d}/*_r315.json")[0])); w = D["w2"]
    return np.array([q["t"] for q in w]), np.array([q["nB"] for q in w])/9216*100
def late(d): t, y = load(d); return y[(t >= 100) & (t <= 150)].mean()
irr = lambda j: 100*(R.RandFabric(96, 0, j).deg != 6).mean()
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), dpi=150, gridspec_kw=dict(width_ratios=[1.1, 1.1, 1]))
a = ax[0]
J1 = [(0.3, "j0.3_s1"), (0.4, "j0.4_s1_g0"), (0.45, "j0.45_s1"), (0.5, "j0.5_s1"), (0.565, "j0.565_s1")]
J2 = [(j, f"tb_j{j:g}") for j in (0.3, 0.45, 0.5, 0.52, 0.54, 0.565, 0.7) if glob.glob(f"{B}/tb_j{j:g}/*_r315.json")]
x1 = [irr(j) for j, _ in J1]; y1 = [late(d) for _, d in J1]; x2 = [irr(j) for j, _ in J2]; y2 = [late(d) for _, d in J2]
a.plot(x2, y2, "-o", color="#1c5cab", lw=2, ms=6, label="стріли гладкі (τ у вузлах ґратки)")
a.plot(x1, y1, "-o", color="#eb6834", lw=2, ms=6, label="стріли із зсунутих точок, як у (315)")
a.axhline(late("gr0_gin0.001_j0.565"), color="#888", ls=":", lw=1.5, label="майже правильна сітка (зсув 0,001)")
a.set_xlabel("неправильних вузлів (≠ 6 сусідів), %"); a.set_ylabel("вузлів у світах, % (t 100–150)")
a.set_title("(а) що гасить світи: граф чи шум стріл", fontsize=10, loc="left"); a.legend(fontsize=7.5, frameon=False); a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
a = ax[1]; ramp = {16: "#9ec5f4", 24: "#3987e5", 32: "#0d366b"}
for g in (16, 24, 32):
    t, y = load(f"gr{g}_gin0.001_j0.565"); n = int(R.RandFabric(96, 0, 0.001).pos.shape[0]*np.pi*g*g/9216/ (np.sqrt(3)/2)) 
    a.plot(t, y, color=ramp[g], lw=2, label=f"зерно R {g}")
a.set_xlabel("час t"); a.set_ylabel("вузлів у світах, %"); a.set_title("(б) бездоганне зерно у склі 13,8 %", fontsize=10, loc="left")
a.legend(fontsize=8, frameon=False); a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
src = open(f"{B}/run316.py").read().split("R.RandFabric.__init__ = _init")[0]; G = {"__name__": "x"}; sys.argv = ["x"]; exec(src, G)
G["GR"] = 32; G["JIN"] = 0.001; R.RandFabric.__init__ = G["_init"]; f = R.RandFabric(96, 0, 0.565)
psi = np.load(glob.glob(f"{B}/gr32_gin0.001_j0.565/*.npz")[0])["psi"]; a = ax[2]
sc = a.scatter(f.pos[:, 0], f.pos[:, 1], c=psi, s=1.2, cmap="Blues", vmin=-1, vmax=1.3, linewidths=0)
b = f.pos[f.deg != 6]; a.scatter(b[:, 0], b[:, 1], s=1.5, c="#eb6834", linewidths=0, label="вузол ≠ 6 сусідів")
a.add_patch(plt.Circle((48, 48), 32, fill=False, ls="--", lw=1.2, color="#333")); a.legend(fontsize=7, loc="lower right", markerscale=4)
a.set_aspect(1); a.set_xticks([]); a.set_yticks([]); a.set_xlim(0, 96); a.set_ylim(0, 96)
a.set_title("(в) зерно R 32 на t 150: світи лише всередині", fontsize=10, loc="left"); fig.colorbar(sc, ax=a, shrink=.8, label="ψ (світ: ψ > 0)")
fig.tight_layout(); fig.savefig("/mnt/project-files/figures/fig211_glass_arrows_grains.png", bbox_inches="tight"); print("ok", list(zip(np.round(x2, 1), np.round(y2, 1))), list(zip(np.round(x1, 1), np.round(y1, 1))))
