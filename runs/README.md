# Runs

Each script runs from any directory and writes into `runs/out/<name>`. All runs restart from their last checkpoint if interrupted (same command).
Single-threaded per run; use `xargs -P <cores>` for several seeds.

| Script | Paper | What |
|---|---|---|
| `birth.sh SEED` | Sec. 4.2, Fig. 3 | model v1, L = 96, s₀ = 3.5, T = 100, snapshots every time unit (173 A) |
| `light.sh PF SEED` | Sec. 4.4, Fig. 5 | v1 with live Φ★, T = 200 (173 B); then `python3 src/ph145.py <..._g91.npz> 0.02` |
| `matter.sh BETA SEED [GV]` | Sec. 4.5, Fig. 6 | v1 with Φ★ and SO(5) field, T = 250; GV = 100 plants a vortex (174) |
| `hole.sh L S0 SEED R Q` | Sec. 4.6, Fig. 7 | a hole of radius R and depth Q cut into the substrate at t = 0, T = 60 (177); R = 0 is the control |
| `dt.sh SEED` | Sec. 3 | as `birth.sh` with dt = 0.0025, no snapshots (175) |
| `window.sh ARGS` | Sec. 4.1, Fig. 2 | budget scan without elasticity (130): `sk=4 ln=0 seed=1 s0=3.0` … |
| `sizeL.sh ARGS` | Sec. 4.3, Fig. 4 | v1 at L = 48–144, T = 250 (69 / 143) |
