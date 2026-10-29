"""Fig. 2 статті: режими запасу. (a) карта ψ моделі v1 при s0 = 3,5 (r173, L96, t = 100); (b) частка у світах; (c) кипіння.
Дані: без пружності — r130/ana130.json (L96, 2 сіди, t ≥ 150); v1 — ela/win (L72, сід 1, t ≥ 100) і ela/sizeL/L72_s1 (s0 = 3,5)."""
import os as _os; _H = _os.path.dirname(_os.path.abspath(__file__)); S = _os.environ.get("NM_DATA", _H + "/../data/")
import sys, json, ast, numpy as np; sys.path.insert(0, _H); from pstyle import *
def load(fn): s = open(fn).read(); i = s.index('{'); return ast.literal_eval(s[i:s.rindex('}') + 1])
r130 = json.load(open(S + "r130/ana130.json"))
v1 = {}
for s0 in ("3.0", "3.3", "3.5", "4.2", "4.5"):
    d = load(S + ("ela/sizeL/L72_s1/log" if s0 == "3.5" else f"ela/win/s{s0}_g0.01/log")); late = [q for q in d["w2"] if q["t"] >= 100]
    v1[float(s0)] = (100*np.mean([q["nB"] for q in late])/72**2, np.mean([q["boil"] for q in late]))
fig = plt.figure(figsize=(12, 3.6)); gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1, 1])
ax = fig.add_subplot(gs[0]); L = 96; psi = np.load(S + "r173/A_s1/psi_t100.npz")["psi"].reshape(L, L)
X, Y = tri_xy(L); ax.scatter(X.ravel(), Y.ravel(), c=np.where(psi.ravel() > 0, V1, "#ecebe7"), s=2.2, marker="h", linewidths=0)
ax.set_aspect("equal"); ax.axis("off"); ax.set_title("model v1, s₀ = 3.5, t = 100: domains (green)", fontsize=9); tag(ax, "a")
for k, (key, lab) in enumerate((("share", "domain fraction (%)"), ("boil", "boiling rate ⟨|Δ|z|²|⟩ per unit time"))):
    ax = fig.add_subplot(gs[k + 1]); tag(ax, "bc"[k])
    for a, b in ((4.2, 6.4),): ax.axvspan(a, b, color=GRAY, alpha=0.25, lw=0)
    ax.axvline(6.854, color="#555", ls="--", lw=1)
    xs = [q["s0"] for q in r130]; ys = [q[key] for q in r130]
    ax.plot(xs, ys, "o", color=NOEL, ms=5, label="without elasticity (L96, 2 seeds)")
    ax.plot(list(v1), [v[k] for v in v1.values()], "D", color=V1, ms=6, label="model v1 (L72, 1 seed)")
    if key == "boil":
        ax.plot([6.5, 6.854], [0, 0], "o", mfc="white", mec="#555", ms=6, label="quiet (L48)")
        ax.text(6.8, 0.25, "w₊²", ha="right", fontsize=8, color="#555")
    else:
        ax.text(5.3, 30, "boiling,\nno domains", ha="center", fontsize=8, color="#555")
    ax.set_xlabel("intensity budget s₀ (mean |z|² per link)"); ax.set_ylabel(lab); ax.set_xlim(2.85, 7.0); ax.legend(fontsize=7, loc="center right" if key == "share" else "upper left", bbox_to_anchor=(1, 0.75) if key == "share" else None)
plt.tight_layout(); save(fig, "fig2_budget")
