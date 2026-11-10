# fig210: (315) аморфна Матриця — частка світів у часі і карти ψ
import json, glob, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/project-files/model/s52/r268/code"); import rnd51 as R
A = json.load(open("/mnt/project-files/model/s52/r315/ana315.json"))
ramp = ["#0d366b", "#1c5cab", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
keys = sorted(A, key=lambda k: (A[k]["jit"], k))
fig = plt.figure(figsize=(13, 4.4), dpi=150); gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])
ax = fig.add_subplot(gs[0]); js = sorted(set(A[k]["jit"] for k in keys)); cmap = {v: ramp[min(i, len(ramp)-1)] for i, v in enumerate(js)}
for k in keys:
    a = A[k]; lab = ("правильна ґратка" if k.startswith("j0_") else f"зсув {a['jit']:g}: {a['irr']:g}".replace(".", ",") + " % неправ.") + (" (друге τ)" if "_s2" in k else "") + (" (сітка 2)" if k.endswith("_g1") else "")
    ax.plot(a["t"], np.array(a["nB"])/9216*100, color=cmap[a["jit"]], lw=2, ls="--" if ("_s2" in k or k.endswith("_g1")) else "-", label=lab)
ax.set_xlabel("час t"); ax.set_ylabel("вузлів у світах, %"); ax.set_title("(а) світи на сітці з неправильними вузлами", fontsize=10, loc="left")
ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=7.5, frameon=False)
for i, (k, j, g, ttl) in enumerate((("j0.3_s1_g0", 0.3, 0, "(б) зсув 0,3, 0 % неправ."), ("j0.45_s1_g0", 0.45, 0, "(в) зсув 0,45, 2,6 % неправ."))):
    if k not in A: k = k.replace("_g0", "")
    f = R.RandFabric(96, g, j); psi = np.load(glob.glob(f"/mnt/project-files/model/s52/r315/{k}/lc_*.npz")[0])["psi"]
    a2 = fig.add_subplot(gs[i+1]); sc = a2.scatter(f.pos[:, 0], f.pos[:, 1], c=psi, s=1.2, cmap="Blues", vmin=-1, vmax=1.3, linewidths=0)
    if j != 0.3: b = f.pos[f.deg != 6]; a2.scatter(b[:, 0], b[:, 1], s=3, c="#eb6834", linewidths=0, label="вузол ≠ 6 сусідів"); a2.legend(fontsize=7, loc="lower right", frameon=True, markerscale=3)
    a2.set_aspect(1); a2.set_xticks([]); a2.set_yticks([]); a2.set_title(ttl + f"\nψ max {psi.max():.2f}".replace(".", ","), fontsize=10, loc="left")
fig.colorbar(sc, ax=fig.axes[1:], shrink=.8, label="ψ (світ: ψ > 0)")
fig.savefig("/mnt/project-files/figures/fig210_amorphous_threshold.png", bbox_inches="tight"); print("ok")
