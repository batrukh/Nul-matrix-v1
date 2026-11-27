"""fig284: (395) багата Матриця s0 3,45 vs s0 3,0 (r374 α0,2, ті самі зерна/стріли τ)."""
import ast, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "/mnt/project-files/model/s52"
def rd(f): return [ast.literal_eval(l) for l in open(f) if l.startswith("{")]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for s in range(1, 9):
    a = rd(f"{S}/r373/a0.2_seed{s}/log") + [r for r in rd(f"{S}/r374/a0.2_seed{s}/log") if r["t"] > 100]; b = rd(f"{S}/r395/runs/s345_seed{s}/log")
    ax[0].plot([r["t"] for r in a], [r["big"]/4096 for r in a], color="tab:blue", lw=1, label="s0 3,0 (373)+(374)" if s == 1 else None)
    ax[0].plot([r["t"] for r in b], [r["big"]/4096 for r in b], color="tab:red", lw=1.4, label="s0 3,45 (395)" if s == 1 else None)
    ax[1].plot([r["t"] for r in a], [r["A2"] for r in a], color="tab:blue", lw=1, label="s0 3,0 (373)+(374)" if s == 1 else None)
    ax[1].plot([r["t"] for r in b], [r["A2"] for r in b], color="tab:red", lw=1.4, label="s0 3,45 (395)" if s == 1 else None)
ax[0].set(xlabel="час, од.", ylabel="найбільший світ, частка ґратки", ylim=(-0.02, 1), title="(а) світ: s0 3,0 — 6/8, s0 3,45 — 0/8")
ax[1].set(xlabel="час, од.", ylabel="середнє |z|² на нитці", title="(б) запас Матриці")
for a_ in ax: a_.legend(fontsize=8)
fig.suptitle("(395) Багата Матриця не народжує світ: тор без меж, Кун L16, α 0,2, T300, ті самі 8 зерен")
fig.tight_layout(); fig.savefig(f"{S}/../../figures/fig284_3d_bagata_matrycia.png", dpi=110)
