"""(320) Релаксація пари при куті φ = PM·π·k/NS із закріпленням ямами. python3 relax.py L R NS k [A w TOL]
Контрольна точка ck/…, результат fin/L{L}_R{R}_N{NS}_k{k}.npz. Загасна динаміка γ 0,5, dt 0,1 (як r312)."""
import numpy as np, sys, os, time, json
sys.path.insert(0, '/mnt/project-files/model/s52/r320'); from pair import *
L, R, NS, k = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
A = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5; w = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0; TOL = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-7
D = '/mnt/project-files/model/s52/r320'; tag = f'L{L}_R{R:g}_A{A:g}_w{w:g}_N{NS}_k{k}'; OUT = f'{D}/fin/{tag}.npz'; CK = f'{D}/ck/{tag}.npz'
os.makedirs(f'{D}/fin', exist_ok=True); os.makedirs(f'{D}/ck', exist_ok=True)
if os.path.exists(OUT): print('є', OUT); sys.exit()
phi = np.pi*k/NS; g = model(L, R, phi, A, w)
if os.path.exists(CK):
    c = np.load(CK); P, Pi, U, E, n, pn, it = (c[x] for x in ['P', 'Pi', 'U', 'E', 'n', 'pn', 'it']); it = int(it)
else:
    P, U, n = ansatz2(L, R, phi); Pi = np.zeros_like(P); E = np.zeros_like(U); pn = np.zeros_like(n); it = 0
tang = lambda v, n: v - np.sum(v*n, -1, keepdims=True)*n
t0 = time.time(); dt, gam = 0.1, 0.5
while True:
    P, Pi, U, E, n, pn = g.step_d(P, Pi, U, E, n, pn, dt, gam); it += 1
    if it % 500 == 0:
        FP, FE, Fn = g.forces_d(P, U, n); fm = max(np.abs(FP).max(), np.abs(FE).max(), np.abs(tang(Fn, n)).max())
        print(tag, it, 'сила %.2e' % fm, round(time.time() - t0), flush=True)
        if fm < TOL or it >= 80000:
            np.savez(OUT, P=P, U=U, n=n, it=it, fm=fm, phi=phi, R=R, A=A, w=w); os.path.exists(CK) and os.remove(CK); break
        if it % 4000 == 0: np.savez(CK, P=P, Pi=Pi, U=U, E=E, n=n, pn=pn, it=it)
