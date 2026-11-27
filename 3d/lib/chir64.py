"""chir64: № 63 крок 2 — чиста хіральність. Відтворює стріли в кінці прогону (τ розпливається детерміновано, від z не залежить),
рахує чисте ⟨sin h⟩ по всіх трикутниках, окремо σ=±1 і «верхні/нижні» трикутники."""
import numpy as np, os, json
os.environ.setdefault("DELTA", "1e-6")
import simNlib as N, rnd51 as R, local60 as A
L = 96
def arrows(seed, D, g, Tt, ell=8, dt=0.005):
    f = N.Fabric(L); tau = A.tau_field(L, ell, seed)
    x_, y_ = f.xy[f.tail].T; x2, y2 = f.xy[f.head].T
    Gg = g*(((x2-x_+L//2) % L-L//2)+0.31*((y2-y_+L//2) % L-L//2))
    bad = tau[f.head]-tau[f.tail]+Gg < 0; R.flip(f, bad); Gg = np.where(bad, -Gg, Gg)
    if D:
        for _ in range(int(round(Tt/(200*dt)))):
            ns = int(np.ceil(12*D))
            for _ in range(ns):
                g_ = tau[f.head]-tau[f.tail]; tau = tau+(D*200*dt/ns)*(np.bincount(f.tail, g_, f.n)-np.bincount(f.head, g_, f.n))
            fl = tau[f.head]-tau[f.tail]+Gg < 0; R.flip(f, fl); Gg = np.where(fl, -Gg, Gg)
    od = np.bincount(f.tail, None, f.n); return f, int(((od == 0) | (od == 6)).sum()), (int((Gg < 0).sum()) if g else None)
def meas(f, z, psi):
    l1, l2, l3 = f.tri_links(z); s = np.sin(np.angle(l1*l2*l3)); sig = np.sign(f.S1+f.S2+f.S3)
    up = np.zeros(f.ntri, bool); up[::2] = True; B = psi > 0; inw = B[f.TV].all(1)
    se = s.std()/np.sqrt(len(s))
    return dict(net=round(float(s.mean()), 5), se=round(float(se), 5), sig=int(sig.sum()), up=round(float(s[up].mean()), 4), dn=round(float(s[~up].mean()), 4),
                lock=round(float((s*sig).mean()), 4), inw=round(float(s[inw].mean()), 4) if inw.any() else None, nin=int(inw.sum()))
cases = [("lc_s03.5_n100_L96_s1_tau8.npz", 1, 0, 0, 200), ("lc_s03.5_n100_L96_s2_tau8.npz", 2, 0, 0, 200),
         ("lc_s03.5_n100_L96_s3_tau8.npz", 3, 0, 0, 200), ("lc_s03.5_n100_L96_s4_tau8.npz", 4, 0, 0, 200),
         ("lc_s03.5_n100_L96_s1_tau8_D1.npz", 1, 1, 0, 200), ("lc_s03.5_n100_L96_s2_tau8_D1.npz", 2, 1, 0, 200),
         ("bak68/lc_s03.5_n100_L96_s1_tau8_D1_g0.02.npz", 1, 1, 0.02, 600), ("bak68/lc_s03.5_n100_L96_s2_tau8_D1_g0.02.npz", 2, 1, 0.02, 600),
         ("lc_s03.5_n100_L96_s1_tau8_D1_g0.01.npz", 1, 1, 0.01, 600), ("lc_s03.5_n100_L96_s1_tau8_D1_g0.04.npz", 1, 1, 0.04, 600),
         ("lc_s03.5_n100_L96_s1_tau8_D1_g0.02.npz", 1, 1, 0.02, 1500), ("lc_s03.5_n100_L96_s2_tau8_D1_g0.02.npz", 2, 1, 0.02, 1500)]
for fn, sd, D, g, Tt in cases:
    f, nx, nd = arrows(sd, D, g, Tt); d = np.load(fn)
    js = fn.replace(".npz", ".json"); last = json.load(open(js))["w2"][-1] if os.path.exists(js) else {}
    print(fn, "T", Tt, "replay ext/def", nx, nd, "json", last.get("next"), last.get("ndef"), meas(f, d["z"], d["psi"]), flush=True)
