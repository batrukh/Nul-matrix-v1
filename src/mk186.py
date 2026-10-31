"""mk186 (перевірка: чи тримає залишок народжень запас |z|): у диску R (типово 9) |z| ниток → випадкові значення з розподілу |z| ниток Матриці поза диском (обидва кінці ψ<0),
решта ниток множиться на сталий множник, щоб Σ|z|² не змінилась (бюджет збережений). Варіанти: amp — лише |z|; ampscr — |z| і фази навмання;
all — |z|, фази навмання і ψ → рівень Матриці. Файли rs/<сід>_k<k>_<варіант>R<R>_e<подія>.npz.
mk183: (183) стани для втручання в острів. Для кожної події (сід, кадр k = j − 3) — диск радіуса R навколо центру нового світу (кадр j).
Варіанти: ctl — без змін (перевірка відтворюваності); scr — фази ниток у диску випадкові (|z| той самий, запас незмінний);
pn — фазовий шум σ = 0,5 рад; sham — те саме scr у диску того ж радіуса в іншому місці Матриці (далі 3 від світів, без народження в кадрах k..k+15).
Пишемо rs/<сід>_k<k>_<варіант>[_<подія>].npz (z, ψ, t) і rs/events.json."""
import numpy as np, json, os, sys
from scipy.ndimage import distance_transform_edt as edt
L = 96; R = float(sys.argv[1]) if len(sys.argv) > 1 else 9.0
SN = os.environ.get("R183_SN", "birth_s{sd}/lc_s03.5_n100_L96_s{sd}_sk4l0_sn_sn.npz")  # кадри birth.sh
ev = [e for e in json.load(open("ev183_dk3.json")) if e["t_j"] <= 22.5]
i_, j_ = np.divmod(np.arange(L*L), L); X = i_ + j_/2; Y = j_*np.sqrt(3)/2            # координати вузлів (i + j/2, j·√3/2)
def tdist(x0, y0):
    best = None
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            d = np.hypot(X - x0 - p*L - q*L/2, Y - y0 - q*np.sqrt(3)/2*L); best = d if best is None else np.minimum(best, d)
    return best
def centre(nodes):                                                                  # центр на торі через кругові середні
    ang = lambda v, P: np.angle(np.mean(np.exp(2j*np.pi*v/P)))*P/(2*np.pi) % P
    jj = ang(j_[nodes], L); ii = ang(i_[nodes], L); return ii + jj/2, jj*np.sqrt(3)/2
from local60 import Loc
f = Loc(L=L, h=0.3, lc=1.3).fab                                                     # sk=4: стріли ґратки, хвости не перевертаються
os.makedirs("rs", exist_ok=True); rng = np.random.default_rng(186 + int(R)); out = []; cache = {}
for n_, e in enumerate(ev):
    sd, k = e["seed"], e["k"]
    if sd not in cache: cache[sd] = np.load(SN.format(sd=sd))
    d = cache[sd]; z = d["z"][k].astype(complex); psi = d["psi"][k].astype(float); t = float(np.round(d["t"][k]))
    x0, y0 = centre(np.array(e["new"])); disk = tdist(x0, y0) <= R
    lk = disk[f.tail] & disk[f.head]
    base = f"{sd}_k{k}"
    pm = float(np.median(psi[(psi < 0) & ~disk])); pr = np.where(disk, pm, psi)                      # ψ у диску → рівень Матриці
    mat = (psi[f.tail] < 0) & (psi[f.head] < 0) & ~lk; am = float(np.mean(np.abs(z[mat])**2))         # |z|² Матриці поза диском
    aa = rng.choice(np.abs(z[mat]), f.m); za = np.where(lk, aa*np.exp(1j*np.angle(z)), z); B = np.sum(np.abs(z)**2)
    sc = np.sqrt((B - np.sum(np.abs(za[lk])**2))/np.sum(np.abs(z[~lk])**2)); za = np.where(lk, za, z*sc)      # бюджет Σ|z|² той самий
    zr = np.where(lk, aa*np.exp(2j*np.pi*rng.random(f.m)), za)
    np.savez(f"rs/{base}_ampR{R:g}_e{n_}.npz", z=za, psi=psi, t=t)
    np.savez(f"rs/{base}_ampscrR{R:g}_e{n_}.npz", z=zr, psi=psi, t=t)
    np.savez(f"rs/{base}_allR{R:g}_e{n_}.npz", z=zr, psi=pr, t=t)
    out.append(dict(ev=n_, seed=sd, k=k, t_k=t, t_j=e["t_j"], nnew=e["nnew"], cx=float(x0), cy=float(y0), R=R, nlinks=int(lk.sum()),
                    pm=pm, am=am, a_disk=float(np.mean(np.abs(z[lk])**2)), a_new=float(np.mean(np.abs(za[lk])**2)), scale=float(sc)))
    print(n_, base, "t", t, "→", e["t_j"], "links", int(lk.sum()), flush=True)
json.dump(out, open(f"rs/events_R{R:g}.json", "w"))
