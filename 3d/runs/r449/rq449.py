"""(449) рішальний тест: те саме поле стріл τ (зерно seed), інший початковий шум. env NS=k: генератор шуму default_rng(seed) → default_rng(seed+1000*k); τ (default_rng(700+seed)) не змінюється. Решта — rq383g без змін."""
import os, sys, numpy as np
NS = int(os.environ["NS"]); SEED = int(dict(a.split("=") for a in sys.argv[1:])["seed"])
_rng = np.random.default_rng
def rng(s=None, *a, **k):
    if isinstance(s, (int, np.integer)) and int(s) == SEED: s = SEED + 1000*NS
    return _rng(s, *a, **k)
AMP = float(os.environ.get("AMP", "1"))                                # (449-контроль) шум старту ×AMP (база 0,01 → 0,01·AMP)
class _Amp:
    def __init__(s, g): s.g = g
    def standard_normal(s, *a, **k): return AMP*s.g.standard_normal(*a, **k)
    def __getattr__(s, k): return getattr(s.g, k)
def rng2(s=None, *a, **k):
    g = rng(s, *a, **k)
    return _Amp(g) if (AMP != 1 and isinstance(s, (int, np.integer)) and int(s) == SEED) else g
np.random.default_rng = rng2
exec(open("/mnt/project-files/model/s52/r396/rq383g.py").read())
