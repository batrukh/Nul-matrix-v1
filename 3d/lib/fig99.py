import json, numpy as np, gauge91 as G, matter98 as M, phi90 as PH
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot([16, 32], [6.45, 10.44], "o-", color="#888", label="без калібрувального поля (98)")
Eg = {d: json.load(open(f"gvort99_L96_d{d}_ka1_kb2_Tr60_g0.json"))[29]["E"] for d in (16, 32)}
ax[0].plot([16, 32], [Eg[16], Eg[32]], "o-", color="#c0623b", label="з калібрувальним полем (99)")
ax[0].set_xlabel("відстань між вихорами d"); ax[0].set_ylabel("енергія пари"); ax[0].set_ylim(0, 12); ax[0].legend()
ax[0].set_title("а) калібрувальне поле робить масу скінченною")
L = 96; z = np.load("gvort99_L96_d32_ka1_kb2_Tr60_g0.npz"); P, U = z["P"], z["U"]
g = G.Gauge(np.full((L, L), -1.0), np.ones((3, L, L)), np.ones((2, L, L)), 1.0, 2.0)
P0 = np.broadcast_to(np.diag([M.U_]*3 + [M.V_]*2), P.shape)
e = g.edens(P, 0*P, U, 0*U) - g.edens(P0, 0*P, np.broadcast_to(np.eye(5), U.shape), 0*U)
X, Y = PH.coords(L)
sc = ax[1].scatter(X.ravel(), Y.ravel(), c=np.clip(e, 0, None).ravel(), s=3, cmap="magma_r")
ax[1].set_xlim(40, 100); ax[1].set_ylim(25, 57); ax[1].set_aspect("equal"); plt.colorbar(sc, ax=ax[1], shrink=.8)
ax[1].set_title("б) два Z₂-вихори після 100 одиниць часу"); ax[1].set_xticks([]); ax[1].set_yticks([])
plt.tight_layout(); plt.savefig("/mnt/project-files/figures/fig13_matter_gauge.png", dpi=130)
