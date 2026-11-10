"""(296) Спін вихору Аліси: кутовий момент обертової хвилі «фотона» довкола вихору, на один квант.
Синтетичний однорідний світ (як r285/r287), база κa=1, κb=2 (c=1), b=0,1. Вихор калібрувально: ланки через відрізок v1–v2 = Zπ.
Хвиля: A = ẑ×∇χ, χ = env(r)·J_ν(kr)·cos(νθ) (θ відносно v1, неперервний у «листі» хвостового вузла ланки), Ȧ з χ̇ = ω·J·sin(νθ).
Вимір: L = Σ r×g, g = −ε·Ȧ×B (ε = √3·κa), H = Σ(edens_B − edens_A) (точно), ω — з фази ν-гармоніки B на кільці.
Передбачення: L·ω/H = ν; довкола вихору дозволені лише ν = m+½ (Z₂: фотон непарний), без вихору — цілі.
python3 spin.py MODE NU ; MODE vort|ctrl ; env K RW T L TR"""
import sys, os, json, time, numpy as np
sys.path.insert(0, '/mnt/project-files/model/s52/r268/code')
import gauge91 as G, phi90 as PH
from gauge91 import DIRS
from scipy.special import jv
mode, NU = sys.argv[1], float(sys.argv[2])
K = float(os.environ.get('K', '0.8')); RW = float(os.environ.get('RW', '24')); T = float(os.environ.get('T', '40'))
L = int(os.environ.get('L', '192')); Tr = float(os.environ.get('TR', '60')); KA, KB = 1.0, 2.0; dt, b = 0.05, 0.1
X, Y = PH.coords(L); a = np.full((L, L), -1.0)
g = G.Gauge(a, np.ones((3, L, L)), np.ones((2, L, L)), KA, KB, b)
P0 = np.diag([-0.3485]*3 + [0.5228]*2); I5 = np.eye(5)
jd = 2*int(0.21*L); i0, j0 = L//2 - 10*L//192, int(0.29*L); cx, cy = float(X[i0, j0]), float(Y[i0, j0])
i1, j1 = i0 - jd//2, j0 + jd; vy2 = float(Y[i1, j1]); xc_ = cx + 0.25
P = np.broadcast_to(P0, (L, L, 5, 5)).copy(); U = np.broadcast_to(I5, (3, L, L, 5, 5)).copy()
Zpi = np.diag([1.0, 1.0, -1.0, -1.0, 1.0]); EV = np.array([(1.0, 0.0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2)])
t0 = time.time()
if mode == 'vort':
    for m, ev in enumerate(EV):
        s_ = (xc_ - X)/ev[0]; yc = Y + s_*ev[1]
        U[m][(s_ > 0) & (s_ < 1) & (yc > cy) & (yc < vy2)] = Zpi
    P = PH.sym0(P + 1e-3*np.random.default_rng(1).standard_normal(P.shape))
    E = np.zeros_like(U); Pi = np.zeros_like(P); t = 0.0
    while t < Tr - 1e-9: P, Pi, U, E = g.step(P, Pi, U, E, dt, gam=1.0); t += dt
E = np.zeros_like(U); Pi = np.zeros_like(P)
print('relaxed', time.time() - t0, flush=True)
TW = mode == 'vort'                                   # антиперіодичний кут лише з вихором
def sheet(x, y):                                      # кут θ' ∈ [0,2π) від напрямку розрізу (+y); лівий берег 0+, правий 2π−
    th = np.mod(np.arctan2(y - cy, x - cx) - np.pi/2, 2*np.pi)
    if TW: th = np.where((y > cy) & (x < xc_) & (th > np.pi), th - 2*np.pi, th)
    return th
def cont(x, y, thref, xr, yr):                        # кут у точці, неперервно продовжений від опорної точки
    d = np.arctan2(y - cy, x - cx) - np.arctan2(yr - cy, xr - cx); return thref + np.angle(np.exp(1j*d))
env = lambda r: np.exp(-(r/RW)**4)
def chi(x, y, th, s):                                 # s=0: J cos, s=1: J sin
    r = np.hypot(x - cx, y - cy); f = env(r)*jv(NU, K*r)
    return f*(np.cos(NU*th) if s == 0 else np.sin(NU*th))
w0 = 2*np.sqrt(KB/(2*KA))*np.sin(K/2)                 # оцінка; справжня ω — з виміру
thn = sheet(X, Y); h = 1e-4; G0 = np.zeros((5, 5)); G0[3, 4], G0[4, 3] = 1, -1
UB = U.copy(); EB = E.copy(); THL = np.zeros((3, L, L)); EL = np.zeros((3, L, L))
for m, ev in enumerate(EV):
    mx, my = X + ev[0]/2, Y + ev[1]/2
    def grad(s):
        th_x1 = cont(mx + h, my, thn, X, Y); th_x0 = cont(mx - h, my, thn, X, Y)
        th_y1 = cont(mx, my + h, thn, X, Y); th_y0 = cont(mx, my - h, thn, X, Y)
        gx = (chi(mx + h, my, th_x1, s) - chi(mx - h, my, th_x0, s))/(2*h)
        gy = (chi(mx, my + h, th_y1, s) - chi(mx, my - h, th_y0, s))/(2*h); return gx, gy
    gx, gy = grad(0); THL[m] = gy*ev[0] - gx*ev[1]                   # A = (∂yχ, −∂xχ)
    gx, gy = grad(1); EL[m] = w0*(gy*ev[0] - gx*ev[1])               # Ȧ з χ̇ = ω·J sin
AMP = float(os.environ.get('AMP', '0.02')); sc = AMP/np.abs(THL).max(); THL *= sc; EL *= sc
for m in range(3):
    c, s = np.cos(THL[m]), np.sin(THL[m]); R = np.broadcast_to(I5, (L, L, 5, 5)).copy()
    R[..., 3, 3] = c; R[..., 4, 4] = c; R[..., 3, 4] = s; R[..., 4, 3] = -s
    UB[m] = R @ U[m]; EB[m] = E[m] + EL[m][..., None, None]*G0
A = [P.copy(), Pi.copy(), U.copy(), E.copy()]; B = [P.copy(), Pi.copy(), UB, EB]
eps = np.sqrt(3)*KA; area = np.sqrt(3)/2; ph = lambda M: 0.5*np.einsum('...ij,ij->...', M, G0)
# кільце для частоти: верхні трикутники, центроїди
tx, ty = X + 0.5, Y + np.sqrt(3)/6; tr_ = np.hypot(tx - cx, ty - cy); tth = cont(tx, ty, thn, X, Y)
ring = (tr_ > 8) & (tr_ < 22)
sh = lambda F, d: np.roll(F, (d[0], d[1]), axis=(0, 1))          # значення з вузла n−d у вузол n
ssg = [ph(np.swapaxes(U[m], -1, -2) @ G0 @ U[m]) for m in range(3)] # перенос фотонної компоненти хвіст→голова (±1)
def meas(A, B):
    e = [ph(B[3][m]) - ph(A[3][m]) for m in range(3)]
    ein = [sh(ssg[m]*e[m], DIRS[m]) for m in range(3)]
    Ad = (1/3)*sum((e[m] + ein[m])[..., None]*EV[m] for m in range(3))          # Ȧ точно у вузлі
    HuA, HdA = g.hol(A[2]); HuB, HdB = g.hol(B[2]); bu = ph(HuB) - ph(HuA); bd = ph(HdB) - ph(HdA)
    d0, d1, d2 = DIRS; d01 = (d0[0] + d1[0], d0[1] + d1[1]); dm2 = (-d2[0], -d2[1])
    bsum = bu + sh(ssg[0]*bu, d0) + sh(ssg[1]*bu, d1)                             # верхні: база n, n−e0, n−e1
    bsum = bsum + sh(bd, d0) + sh(ssg[2]*sh(bd, d0), d2) + sh(ssg[1]*sh(bd, d0), d1)
    Bn = bsum/6/(np.sqrt(3)/4)
    ex, ey = X - cx, Y - cy
    gx, gy = -eps*Ad[..., 1]*Bn, eps*Ad[..., 0]*Bn
    far = np.hypot(ex, ey) > 5
    Lz = area*(ex*gy - ey*gx); Hc = area*(0.5*eps*np.sum(Ad**2, -1) + 0.5*(np.sqrt(3)/2*KB)*Bn**2)
    H = float((g.edens(*B) - g.edens(*A)).sum())
    FA = g.forces(A[0], A[2])[1]; FB = g.forces(B[0], B[2])[1]
    f = [ph(FB[m]) - ph(FA[m]) for m in range(3)]
    w2 = sum((f[m]**2).sum() for m in range(3))/sum((e[m]**2).sum() for m in range(3))
    Bf = bu/(np.sqrt(3)/4); Zr = complex(np.sum(Bf[ring]*np.exp(-1j*NU*tth[ring])))
    return dict(L=float(Lz.sum()), Lfar=float(Lz[far].sum()), H=H, Hc=float(Hc.sum()), w=float(np.sqrt(w2)), Zr=Zr)
rec = []; t = 0.0
while True:
    if abs(t - round(t)) < dt/2:
        q = meas(A, B); q['t'] = round(t, 2); q['Zr'] = [q['Zr'].real, q['Zr'].imag]; rec.append(q); print(q, flush=True)
    if t > T - 1e-9: break
    A = list(g.step(*A, dt)); B = list(g.step(*B, dt)); t += dt
ph_ = np.unwrap(np.angle([complex(*q['Zr']) for q in rec])); tt = np.array([q['t'] for q in rec])
om = -np.polyfit(tt[:21], ph_[:21], 1)[0]
Lm = np.mean([q['L'] for q in rec]); Hm = np.mean([q['H'] for q in rec])
wm = np.mean([q['w'] for q in rec]); Hcm = np.mean([q['Hc'] for q in rec]); res = dict(Hcmean=Hcm, ratio_wc=Lm*wm/Hcm, wrms=wm, ratio_w=Lm*wm/Hm, mode=mode, nu=NU, K=K, RW=RW, L=L, T=T, w0=w0, omega=om, Lmean=Lm, Hmean=Hm, ratio=Lm*om/Hm, rec=rec, sec=time.time() - t0, v1=[cx, cy], v2=[cx, vy2])
print('RESULT', mode, NU, 'omega', om, 'L*w/H', Lm*om/Hm, 'wrms', wm, 'L*wrms/H', Lm*wm/Hm, flush=True)
json.dump(res, open(f'spin_{mode}_nu{NU:g}' + os.environ.get('TAG', '') + '.json', 'w'))
