"""psi101: речовина, крок 4 — чи може Z₂-вихор відчути анізотропію ниток. Вихор (протополе + калібрувальне поле) зв'язаний з нитками лише через ψ:
a(x) = 1 − 2χ(ψ). Міряємо в однорідному світі w64 з нахилом τ (α = 1, нитки анізотропні на 23 %) і в контролі α = 0:
ψ (середнє, розкид), δa = a − ⟨a⟩, анізотропію кореляцій δψ уздовж трьох напрямків сітки після T = 60."""
import numpy as np, json, sys, w64
def run(ac=1.0, L=96, s0=8.0, T=60.0, dt=0.005, seed=1):
    s, z, psi = w64.build(L, s0, ac, tilt=1, seed=seed); t = 0.0
    while t < T - 1e-9: z, psi = s.step(z, psi, dt); t += dt
    p = psi.reshape(L, L); chi = 0.5*(1 + np.tanh(p/0.2)); a = 1 - 2*chi; dp = p - p.mean()
    S = [float(np.mean((np.roll(dp, d, axis=(0, 1)) - dp)**2)) for d in ((1, 0), (0, 1), (-1, 1))]
    out = dict(ac=ac, psi_mean=float(p.mean()), psi_sd=float(p.std()), a_mean=float(a.mean()), a_sd=float(a.std()), a_range=float(a.max() - a.min()), S=S,
               aniso=float((max(S) - min(S))/np.mean(S)) if np.mean(S) > 0 else 0.0)
    print(out, flush=True); json.dump(out, open(f"psi101_a{ac:g}.json", "w")); return out
if __name__ == "__main__":
    run(ac=float(sys.argv[1]))
