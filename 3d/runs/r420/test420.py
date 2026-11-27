"""Перевірка qforce3 (420): (1) ph ≡ 1 — тотожно qforce2 на ґратці Куна L16; (2) фаза на всіх петлях = калібрування:
ph_q = conj(u_{e0}) для петлі, де нитка e0 (за знаком) множиться на u — сила qforce3 на z має збігатися з qforce2 на z' (z'_{e0}=z_{e0}·u) з поправкою conj(u);
(3) градієнт: скінченна різниця ціни петель W = Σ cost по ψ_v проти fc (обидва набори)."""
import os, sys, numpy as np
R_ = "/mnt/project-files/model/s52"; sys.path[:0] = [R_ + "/r420", R_ + "/r351", R_]; os.environ["NB"] = "0"; os.environ.setdefault("DELTA", "1e-6")
import qf2, qf3, rq351f as Q
s, f, mod, z, psi = Q.run(L=16, seed=3, s0=3.0, ac=0.1, check=1)
rng = np.random.default_rng(1); z = z*np.exp(1j*rng.uniform(-1, 1, f.m)); psi = rng.uniform(-1, 1, f.n)
E2 = np.ascontiguousarray(np.array(mod.E)); S2 = np.ascontiguousarray(np.array(mod.S)); QV = np.ascontiguousarray(mod.QV)
chi = ((1 + psi)/2)**2; dchi = (1 + psi)/2; args = (float(mod.eps0), mod.d2, mod.a, f.m, f.n)
a2 = qf2.qforce2(z, E2, S2, QV, chi, dchi, *args); a3 = qf3.qforce3(z, E2, S2, QV, np.ones(len(QV), complex), chi, dchi, *args)
print("(1) ph=1: max відн. різниця", max(np.abs(x - y).max()/np.abs(x).max() for x, y in zip(a2, a3)))
# (2) одна петля q0 з фазою ph проти qforce2 з першою ниткою, повернутою на ph (за знаком)
q0 = 123; Eq = np.ascontiguousarray(E2[:, q0:q0+1]); Sq = np.ascontiguousarray(S2[:, q0:q0+1]); Qq = np.ascontiguousarray(QV[q0:q0+1]); ph = np.exp(0.7j)
b3 = qf3.qforce3(z, Eq, Sq, Qq, np.array([ph]), chi, dchi, *args)
z2 = z.copy(); e0 = Eq[0, 0]; z2[e0] = z[e0]*(ph if Sq[0, 0] > 0 else np.conj(ph)); b2 = qf2.qforce2(z2, Eq, Sq, Qq, chi, dchi, *args)
u = ph if Sq[0, 0] > 0 else np.conj(ph)
err = [abs(b3[0][e0] - np.conj(u)*b2[0][e0]), abs(b3[1][e0] - np.conj(u)*b2[1][e0])] + [abs(b3[0][e] - b2[0][e]) + abs(b3[1][e] - b2[1][e]) for e in Eq[1:, 0]] + [np.abs(b3[2] - b2[2]).max()]
print("(2) одна петля з фазою 0,7: max похибка", max(err), "масштаб", abs(b3[0][e0]))
# (3) fc = ∂/∂ψ Σ_q cost_q·chi? — у ядрі fc[v] += dchi[v]*cost_q, де Et залежить від chi лише в cE; перевіримо, що cost не залежить від ψ (тоді fc — точний множник)
PH = np.exp(1j*rng.uniform(0, 2*np.pi, len(QV)))
c1 = qf3.qforce3(z, E2, S2, QV, PH, chi, dchi, *args)[2]; c2 = qf3.qforce3(z, E2, S2, QV, PH, chi*0 + 0.3, dchi, *args)[2]
print("(3) fc не залежить від chi (лише від dchi):", np.abs(c1 - c2).max())
# ціна пласкої петлі = 0, випадкової фази > 0
print("(4) середня ціна на вузол: пласкі ph=1", qf3.qforce3(np.full(f.m, np.sqrt(3.0)+0j), E2, S2, QV, np.ones(len(QV), complex), chi, np.ones(f.n), *args)[2].mean(),
      " випадкові фази", qf3.qforce3(np.full(f.m, np.sqrt(3.0)+0j), E2, S2, QV, PH, chi, np.ones(f.n), *args)[2].mean())
