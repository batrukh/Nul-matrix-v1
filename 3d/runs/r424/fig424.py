import ast, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for k, (arm, ttl) in enumerate([("A", "А: α 0,1, h 0,24 (середнє ψ −0,86)\nпередбачення «світу нема» — 8/8 зі світом"), ("B", "Б: α 0,25, h 0,36 (середнє ψ −0,82)\nпередбачення «світ є» — 4/8")]):
    for s in range(1, 9):
        rows = [ast.literal_eval(l) for l in open(f"runs/{arm}_seed{s}/log") if l.startswith("{")]
        ax[k].plot([r["t"] for r in rows], [100*r["big"]/4096 for r in rows], lw=1.3, label=f"зерно {s}")
    ax[k].set_title(ttl, fontsize=10); ax[k].set_xlabel("час t"); ax[k].grid(alpha=0.3); ax[k].set_yscale("symlog", linthresh=1)
ax[0].set_ylabel("найбільший світ, % ґратки"); ax[1].legend(fontsize=7, ncol=2)
ax[1].text(150, 12, "сталі острови в живій Матриці\n(зерна 5, 6, 7)", fontsize=8)
plt.suptitle("(424) Гіпотезу глибини Матриці спростовано", fontsize=11); plt.tight_layout()
plt.savefig("fig294.png", dpi=150); import shutil; shutil.copy("fig294.png", "/mnt/project-files/figures/fig294_hlybyna_matryci.png")
