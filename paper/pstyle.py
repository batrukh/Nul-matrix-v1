"""Єдиний стиль малюнків компактної статті (англ. підписи): шрифт 9 pt, панелі (a), (b)..., два кольори моделей, PDF + PNG."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
V1, NOEL, GRAY, INK = "#1b7f6a", "#8a8a8a", "#c9c8c3", "#222222"     # v1 (з пружністю) — зелений; без пружності — сірий
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555", "legend.frameon": False,
                     "savefig.bbox": "tight", "pdf.fonttype": 42})
import os as _os; OUT = _os.path.dirname(_os.path.abspath(__file__)) + "/../figures/"
def tag(ax, s): ax.text(-0.12, 1.04, f"({s})", transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
def save(fig, name):
    import os; os.makedirs(OUT, exist_ok=True); fig.savefig(OUT + name + ".pdf"); fig.savefig(OUT + name + ".png", dpi=160)
def tri_xy(L):                                                     # декартові координати вузлів трикутної ґратки (i + j/2, j·√3/2)
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij"); return (i + j/2) % L, j*np.sqrt(3)/2
