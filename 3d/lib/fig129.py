import json, glob, numpy as np, matplotlib, matplotlib.ticker; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import ana53 as A
B, O, G3, INK, MUT = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUT, "xtick.color": MUT, "ytick.color": MUT, "axes.spines.top": False, "axes.spines.right": False})
c = lambda x: f"{x:g}".replace(".", ",")
fig = plt.figure(figsize=(15, 8.6)); gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1])
L = 96
maps = [("sk4l0", "без швів (контроль)"), ("sk1l12", "один шов ℓ = 12"), ("sk2l48", "хрест відрізків ℓ = 48"), ("sk3l48", "замкнена петля 48×48")]
for j, (tag, ttl) in enumerate(maps):
    ax = fig.add_subplot(gs[0, j]); nm = f"lc_s03.5_n100_L96_s1_{tag}"
    sp = np.load(nm + "_sp.npz"); psi = sp["psi"][-1]; bad = np.load(nm + ".npz")["bad"]; f = A.fabric(bad)
    X, Y = A.cart(f.xy[:, 0].astype(float), f.xy[:, 1].astype(float))
    ax.scatter(X, Y, c=np.where(psi > 0, B, "#e8e7e3"), s=3.2, marker="h", linewidths=0)
    for e in np.where(bad)[0]:
        a, b_ = f.tail[e], f.head[e]
        if abs(X[a] - X[b_]) < 3 and abs(Y[a] - Y[b_]) < 3: ax.plot([X[a], X[b_]], [Y[a], Y[b_]], color=O, lw=1.6)
    S = np.stack([f.S1, f.S2, f.S3], 1)
    ch = [(X[f.TV[i]].mean(), Y[f.TV[i]].mean(), int(np.sign(S[i].sum()))) for i in np.where(np.abs(S.sum(1)) == 3)[0]]
    used = set()
    for a_, (tx, ty, sg) in enumerate(ch):
        if a_ in used: continue
        pair = [b2 for b2, q in enumerate(ch) if b2 != a_ and np.hypot(q[0] - tx, q[1] - ty) < 3]
        used.update(pair); txt = "±" if pair else ("+" if sg > 0 else "−")
        ax.plot(tx, ty, "o", ms=11, mfc="white", mec=INK, mew=1.2); ax.text(tx, ty, txt, ha="center", va="center", fontsize=8, color=INK)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(("а) " if j == 0 else "") + ttl + f", t = 250", fontsize=9)
ax0 = fig.axes[0]; ax0.text(0, -7, "синє — світ (ψ > 0), помаранчеве — шов, ○ — петля до-часу (заряд ±1)", fontsize=8, color=MUT)
r = sorted(json.load(open("ana53.json")), key=lambda q: (q["sk"] != 4, q["sk"], q["ln"], q["seed"]))
cfg = []
for q in r:
    k = (q["sk"], q["ln"])
    if k not in cfg: cfg.append(k)
name = {1: "шов", 2: "хрест", 3: "петля", 4: "без швів"}
lab = [name[s] + ("" if s == 4 else f" {l}") for s, l in cfg]
ax = fig.add_subplot(gs[1, 0:2])
for q in r:
    x = cfg.index((q["sk"], q["ln"])) + (q["seed"] - 1.5)*0.18
    ax.plot(x, 100*q["nB"]/L**2, "o", ms=7, color=B if q["seed"] == 1 else O, mec="white", mew=1)
ctl = [100*q["nB"]/L**2 for q in r if q["sk"] == 4]; ax.axhspan(min(ctl), max(ctl), color=G3, alpha=0.18, lw=0)
ax.set_xticks(range(len(cfg))); ax.set_xticklabels(lab, rotation=35, ha="right"); ax.set_ylabel("частка Матриці у світах, %, t ≥ 150")
ax.plot([], [], "o", color=B, label="реалізація 1"); ax.plot([], [], "o", color=O, label="реалізація 2"); ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_ylim(0, 50); ax.set_title("б) світи є за будь-яких швів і без них; сіра смуга — контроль без швів", fontsize=9)
ax = fig.add_subplot(gs[1, 2:4]); on = sorted([json.load(open(p)) for p in glob.glob("nz53/onset_*.json")], key=lambda d: d["s0"])
xs = [d["s0"] for d in on if d["onset"] is not None]; ys = [d["onset"] for d in on if d["onset"] is not None]
ax.plot(xs, ys, "-o", color=B, lw=2, ms=7, mec="white", mew=1)
for d in on:
    if d["onset"] is None: ax.plot(d["s0"], d["Tt"], "^", color=G3, ms=9); ax.annotate(f"не закипає до t = {c(d['Tt'])}", (d["s0"], d["Tt"]), xytext=(-105, -14), textcoords="offset points", fontsize=7, color=MUT)
ax.axvline(3.5, color=O, lw=1.2, ls="--"); ax.text(3.55, max(ys)*0.9, "базовий запас 3,5", color=O, fontsize=7)
ax.axvline(6.86, color=G3, lw=1.2, ls="--"); ax.text(6.82, 60, "природний 6,86", color=MUT, fontsize=7, ha="right")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: c(round(v, 2))))
ax.set_xlabel("запас натягу s₀ (середній |z|² на нитку)"); ax.set_ylabel("час до закипання (кипіння > 0,1)")
ax.set_title("в) упорядкована Матриця без жодного дефекту, шум 10⁻⁶: закипає сама при запасі ≲ 6", fontsize=9)
plt.tight_layout(); plt.savefig("fig41_seams.png", dpi=130)
