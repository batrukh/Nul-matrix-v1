import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for d, c in ((16, "#4a7bb7"), (32, "#c0623b")):
    r = json.load(open(f"matter98_vortex_L96_lam4_d{d}_g0_Tr5.json")); t = [q["t"] for q in r]
    s = [abs(q["cl"][1][0] - q["cl"][0][0]) if q["ncl"] == 2 else 0 for q in r]
    ax[0].plot(t, s, "o-", ms=3, color=c, label=f"старт d = {d}")
ax[0].set_xlim(0, 40); ax[0].set_xlabel("час"); ax[0].set_ylabel("відстань між вихорами"); ax[0].legend()
ax[0].set_title("а) пара Z₂-вихорів притягується і знищується")
for lam, c, f in ((4, "#888", "matter98_lump_L64_lam4_d16_g0.json"), (8, "#4a7bb7", "matter98_lump_L96_lam8_d16_g0_Tr0.json"), (16, "#c0623b", "matter98_lump_L96_lam16_d16_g0_Tr0.json")):
    r = json.load(open(f)); r = [q for q in r if q["t"] <= 20]
    ax[1].plot([q["t"] for q in r], [abs(q["Ql"]) for q in r], "o-", ms=3, color=c, label=f"згусток λ = {lam}")
ax[1].set_xlabel("час"); ax[1].set_ylabel("|заряд Q|"); ax[1].legend(); ax[1].set_title("б) згусток стискається і втрачає заряд")
plt.tight_layout(); plt.savefig("/mnt/project-files/figures/fig12_matter_bare.png", dpi=130)
