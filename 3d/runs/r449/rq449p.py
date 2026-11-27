"""(449) контроль 2.10, виправлений: шум старту ×AMP ЛИШЕ в ψ. У rq449.py AMP множив і шум |z| → середнє |z|² росло на 2σ² (s0 3,0 → ≈3,06, A2 3,077) і світ не народжувався зовсім — збій контролю. Тут z-шум базовий (0,01), ψ-шум ×AMP. Порядок викликів у rq351f: 1–2 — z (розмір m), 3 — ψ (розмір n)."""
import os, sys, numpy as np
NS = int(os.environ["NS"]); SEED = int(dict(a.split("=") for a in sys.argv[1:])["seed"]); AMP = float(os.environ.get("AMP", "1"))
_rng = np.random.default_rng
class _AmpPsi:
    def __init__(s, g): s.g = g; s.c = 0
    def standard_normal(s, *a, **k):
        s.c += 1; x = s.g.standard_normal(*a, **k)
        return AMP*x if s.c == 3 else x
    def __getattr__(s, k): return getattr(s.g, k)
def rng(s=None, *a, **k):
    if isinstance(s, (int, np.integer)) and int(s) == SEED: return _AmpPsi(_rng(SEED + 1000*NS, *a, **k))
    return _rng(s, *a, **k)
np.random.default_rng = rng
exec(open("/mnt/project-files/model/s52/r396/rq383g.py").read())
