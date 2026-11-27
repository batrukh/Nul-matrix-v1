"""(464Б) Комбінаторна кривина K_v = (π/6)(14 − z_v) для простих дефектів ОЦК: вакансія, міжвузля (октаедричне й тетраедричне), пара Френкеля. Повна кривина в одиницях π/6: Σ(14 − z)."""
import sys, numpy as np
sys.path[:0] = ["/mnt/project-files/model/s52/r447", "/mnt/project-files/model/s52"]; import fabpts as F
L = 6; g = np.indices((L, L, L)).reshape(3, -1).T.astype(float); P0 = np.concatenate([g, g + 0.5])
rng = np.random.default_rng(0); P0 = P0 + 1e-6*rng.standard_normal(P0.shape)
def tot(P, lab):
    f = F.fromPoints(P % L, float(L)); z = np.asarray(f.deg); n = f.n; T = len(f.tets)
    print(f"{lab:28s} N {n}  Σ(14−z) = {int((14 - z).sum()):+d}  check 2(6N−T) = {2*(6*n - T):+d}  z: {np.bincount(z)[10:20] if z.max()<20 else z.max()}")
tot(P0, "ідеальна ОЦК")
tot(np.delete(P0, 0, 0), "вакансія")
tot(np.vstack([P0, [[0.5, 0, 0]]]), "міжвузля октаедричне")
tot(np.vstack([P0, [[0.5, 0.25, 0]]]), "міжвузля тетраедричне")
tot(np.vstack([np.delete(P0, 0, 0), [[3.5, 3.0, 3.0]]]), "пара Френкеля (далеко)")
tot(np.vstack([np.delete(P0, 0, 0), np.delete(P0, 0, 0)[:0], [[3.5, 3.0, 3.0], [1.5, 1.0, 1.0]]]), "вакансія + 2 міжвузля")
