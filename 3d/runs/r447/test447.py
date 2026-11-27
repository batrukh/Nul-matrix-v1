"""Перевірка модуля: rq447 lat=kun (сусіди/петлі з тетраедрів, K = P^n множенням) проти rq383g sgn=0 (Кун FabricG, K через FFT) — той самий стан, 1 од. часу."""
import os, sys, numpy as np
os.environ["NB"] = "0"
import rq447 as W, rq351f as Q, fabgen as G, local60 as LC, rnd51 as RN, simNlib as N
ac, s0, seed = 0.1, 3.0, 3
sP, fP, mP, zP, psP = W.run(lat="kun", seed=seed, ac=ac, s0=s0, check=1)
sG, fG, mG, zG, psG = Q.run(L=16, seed=seed, s0=s0, ell=4.0, ac=ac, check=1)
print("Кун: ниток", fG.m, fP.m, "трикутників", fG.ntri, fP.ntri, "петель", len(mG.QV), len(mP.QV))
key = lambda f: np.minimum(f.tail, f.head).astype(np.int64)*f.n + np.maximum(f.tail, f.head)
kG, kP = key(fG), key(fP); o = np.argsort(kP); mp = o[np.searchsorted(kP[o], kG)]; assert (kP[mp] == kG).all()
print("орієнтація ниток після стріл збігається:", bool((fG.tail == fP.tail[mp]).all()))
print("β на нитку збігається:", np.allclose(mG.beta, mP.beta[mp]), " ε0", mG.eps0, mP.eps0, " κ", mG.kappa, mP.kappa, " λc", sG.lc, sP.lc)
zP = np.empty_like(zG); zP[mp] = zG; psP = psG.copy()
for s, z in ((sG, zG), (sP, zP)): s.S0 = float((np.abs(z)**2).sum()); s.m0 = s.S0/len(z)
for i in range(200):
    zG, psG = sG.step(zG, psG, 0.005); zP, psP = sP.step(zP, psP, 0.005)
print("t=1: max|Δz|", float(np.abs(zG - zP[mp]).max()), " max|Δψ|", float(np.abs(psG - psP).max()), " |z| масштаб", float(np.abs(zG).mean()))
