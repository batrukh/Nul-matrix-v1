import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for d, c in ((16, "#c0623b"), (20, "#4a7bb7"), (24, "#888")):
    r = [q for q in json.load(open(f"gvort99_L96_d{d}_ka1_kb2_Tr60_g0.json")) if q["t"] >= 60 and q["W"][0] < 0]
    ax[0].plot([q["t"] - 60 for q in r], [q["pos"][1] - q["pos"][0] for q in r], color=c, label=f"старт d = {d}")
ax[0].set_xlabel("час"); ax[0].set_ylabel("відстань між вихорами"); ax[0].legend(); ax[0].set_title("а) вихори падають повільніше за світло (v ≤ 0,63)")
for f, lab, c in (("photon100_b0_vac1", "без вихору", "#888"), ("photon100_b6_vac0", "повз вихор, b = 6", "#4a7bb7"), ("photon100_b0_vac0", "прямо у вихор, b = 0", "#c0623b")):
    r = json.load(open(f + ".json"))["rec"]; ax[1].plot([q["t"] for q in r], [q["fwd"]*100 for q in r], color=c, label=lab)
ax[1].set_xlabel("час"); ax[1].set_ylabel("частка енергії «фотона» за вихором, %"); ax[1].legend(); ax[1].set_title("б) вихор розсіює/поглинає світло")
plt.tight_layout(); plt.savefig("/mnt/project-files/figures/fig14_matter_motion_light.png", dpi=130)
