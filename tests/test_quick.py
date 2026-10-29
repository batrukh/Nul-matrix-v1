"""Quick check (~30 s): (1) the compiled force equals the plain numpy force; (2) model v1 at L = 48 boils by itself
(the ordered network becomes disordered without noise) with the statistics of the paper. Run: python3 tests/test_quick.py"""
import os, sys, json, subprocess, tempfile
H = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(H, "..", "src"); sys.path.insert(0, SRC)
import numpy as np, stall43 as T, rnd51 as R, seam51 as SM, nbforce as NB
s = T.Sys(L=24, h=0.3, lc=1.3); f = s.fab; R.flip(f, SM.seams(f)); rng = np.random.default_rng(3)
z = np.sqrt(3.5)*(1 + 0.3*(rng.standard_normal(f.m) + 1j*rng.standard_normal(f.m))); psi = rng.uniform(-1.3, 1.2, f.n); s.m0 = 3.4
a = T.Sys.F(s, z, psi); b = NB.force(s, z, psi)
e = max(np.abs(a[0] - b[0]).max()/np.abs(a[0]).max(), np.abs(a[1] - b[1]).max()/np.abs(a[1]).max())
assert e < 1e-10, f"compiled force differs: {e}"; print("force: compiled = numpy, rel. error %.1e" % e)
with tempfile.TemporaryDirectory() as d:
    subprocess.run([sys.executable, os.path.join(SRC, "run145.py"), "L=48", "n=100", "s0=3.5", "Tt=20", "sk=4", "ln=0", "seed=1"], cwd=d, check=True, capture_output=True)
    w = json.load(open(os.path.join(d, "lc_s03.5_n100_L48_s1_sk4l0_eg0.01.json")))["w2"]; w = w if isinstance(w, dict) else w[-1]
print("t = 20: mean |z|^2 %.3f, spread %.3f, loop coherence %.3f, boiling rate %.3f" % (w["A2"], w["sd"], w["hol"], w["boil"]))
assert abs(w["A2"] - 3.5) < 0.1 and 2.5 < w["sd"] < 3.3 and 0.6 < w["hol"] < 0.9 and w["boil"] > 0.1, "statistics outside the expected range"
print("OK")
