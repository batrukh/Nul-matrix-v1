import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for f, lab, c in (("real102_world_f0.02", "світ глибини 8 (L144)", "#c0623b"), ("real102_L96_b1.2_world_f0.02", "світ глибини 13 (β = 1,2)", "#4a7bb7")):
    r = json.load(open(f + ".json"))["rec"]; t = [q["t"] for q in r]
    ax[0].plot(t, [q["W"][1] for q in r], color=c, label=lab); ax[1].plot(t, [q["inM"]*100 for q in r], color=c, label=lab)
for a_ in ax: a_.axvline(40, color="#aaa", ls="--", lw=1); a_.set_xlabel("час (до 40 — релаксація з тертям)"); a_.legend()
ax[0].set_ylabel("заряд у петлі r = 5 (−1 — вихор на місці)"); ax[0].set_title("а) вихор лишає місце: швидко в мілкому світі")
ax[1].set_ylabel("частка енергії вихору в Матриці, %"); ax[1].set_title("б) у мілкому світі розчиняється в Матриці")
plt.tight_layout(); plt.savefig("/mnt/project-files/figures/fig16_matter_realworld.png", dpi=130)
