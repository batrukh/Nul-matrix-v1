"""(347) крок 4: крапля світу (φ⁴-стінка, як (345)) як джерело. Енергія у вузлі E_v = e(r_v)·A_v⁰ (густина стінки ½ψ'² + ¼(1−ψ²)², ψ = tanh((r0 − r)/w)),
δ_v = 8πG·E_v; 8πG обрано так, щоб Σδ = 40°. Перевірки: (а) Σδ; (б) площа геодезичного диска A(r) (Дейкстра по довжинах ниток, ділена на пласку)
→ всередині 1, зовні (1 − δ/2π)·r² + стала; (в) лінза хвилею, як lens347 (фронти позаду, r 45–80)."""
import sys, json, numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, '/mnt/project-files/model/s52/r347'); from regge347 import *; from lens347 import front_dir
R, r0, w = 90, 15.0, 2.0; DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
P, T, E, inner = disk(R); n = len(P); rr = np.hypot(*P.T); c0 = int(np.argmin(rr))
psi_p = -(1/w)/np.cosh((r0 - rr)/w)**2; psi = np.tanh((r0 - rr)/w)
e = 0.5*psi_p**2 + 0.25*(1 - psi**2)**2
A0 = np.full(n, np.sqrt(3)/2)                 # дуальна площа вузла пласкої ґратки
Ev = e*A0; Ev[~inner] = 0; G8 = np.radians(DEG)/Ev.sum()
delta = G8*Ev; u, err, it = solve(P, T, inner, delta)
L0 = tri_L0(P, T); dsum = 2*np.pi - angsum(L0, u, T, n)
out = dict(r0=r0, w=w, Etot=float(Ev.sum()), G8=float(G8), err=float(err), it=int(it), total_deficit=float(np.degrees(dsum[inner].sum())),
           sigma_theory=float(2*np.sqrt(2)/3/w*w), max_node_deficit=float(np.degrees(delta.max())))
# (в) світло
rt = np.hypot(*P[T].mean(1).T); allow = rt > r0 + 5*w                 # розгортка лише пласкої зовнішньої частини (кільце з розрізом позаду)
Q = layout(P, T, u, allow=allow, start=(-40.0, 0.3))
# центр розгортки: апекс зовнішнього конуса — точка, від якої відстані до вузлів кола однакові (МНК)
ring_ = np.isfinite(Q[:, 0]) & (rr > 30) & (rr < 34)
A_ = np.c_[2*Q[ring_], np.ones(ring_.sum())]; cxy = np.linalg.lstsq(A_, (Q[ring_]**2).sum(1), rcond=None)[0][:2]; Q = Q - cxy; K, M, area = cot_lap(P, T, u)
Q = np.where(np.isfinite(Q), Q, 0.0)*np.isfinite(Q) + np.where(np.isfinite(Q), 0, np.nan); x = np.nan_to_num(Q[:, 0], nan=0.0); x0 = -50.0; g = np.exp(-((x - x0)/3.0)**2); left = (x < -28) & (rr < R - 5) & np.isfinite(x)
f = np.where(left, g, 0.0); v = np.where(left, 2*(x - x0)/9.0*g, 0.0); fix = rr >= R - 1.5; tarr = np.full(n, np.nan); fmax = np.zeros(n)
acc = -(K @ f)/M; t = 0.0; dt = 0.2; snap = None
while t < 150.0:
    v += 0.5*dt*acc; f += dt*v; f[fix] = 0; acc = -(K @ f)/M; v += 0.5*dt*acc; t += dt
    new = f > fmax; fmax = np.where(new, f, fmax); tarr = np.where(new & (f > 0.15), t, tarr)
    if abs(t - 100.0) < dt/2: snap = f.copy()
ang = np.degrees(np.arctan2(Q[:, 1], Q[:, 0])); r = np.hypot(*Q.T); lo = DEG/2 + 8
up = (r > 45) & (r < 80) & (ang > lo) & (ang < lo + 40); dn = (r > 45) & (r < 80) & (ang < -lo) & (ang > -lo - 40)
a_up, v_up = front_dir(Q, tarr, up); a_dn, v_dn = front_dir(Q, tarr, dn)
a360 = np.mod(ang[(r > 30) & (r < 60)], 360)
out.update(dir_up=a_up, dir_dn=a_dn, lens=DEG + (a_up - a_dn), v_up=v_up, v_dn=v_dn, span=float(a360.max() - a360.min()))
print({k: v for k, v in out.items() if not isinstance(v, list)}, flush=True)
np.savez(f'/mnt/project-files/model/s52/r347/drop347_d{int(DEG)}.npz', P=P, Q=Q, u=u, delta=delta, snap=snap, tarr=tarr)
json.dump(out, open(f'/mnt/project-files/model/s52/r347/drop347_d{int(DEG)}.json', 'w'), indent=1)
