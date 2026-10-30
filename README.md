# Null Matrix (Nul-matrix-v1), model v1

Code and data for the paper on model v1: a triangular network of complex links with a conserved intensity budget that boils by itself,
nucleates domains ("worlds"), keeps them apart by an elastic response of the substrate, and hosts a confined massless gauge mode and
Z₂ vortices. The paper also reports negative results (no long-range gravity, no domains in 3D), and their code is included.

Comments in the source are in Ukrainian; names of quantities follow the paper.

## Install and check

```
pip install -r requirements.txt
python3 tests/test_quick.py        # ~30 s: compiled force = numpy force; model boils at L = 48
```

## Redraw the figures (no simulation needed)

```
for f in paper/fig*.py; do python3 $f; done     # writes figures/*.pdf and *.png from data/
```

`data/` holds only the files the figures read (5 MB). Set `NM_DATA` to use another data folder, e.g. your own `runs/out`.

## Reproduce the numbers

Scripts in `runs/` (see `runs/README.md`). Reference parameters of model v1: L = 96, n = 100, s₀ = 3.5, β = 1.5, λ_c = 1.3, ε₀ = 3, h = 0.3,
κ = 1, elastic ℓ = 8, γ = 0.01, lattice arrows (`sk=4 ln=0`), RK4 with dt = 0.005. Times are wall-clock on 4 cores for all seeds of a row where measured; — means not measured.

| Paper | Number(s) | Run | Analysis | Time |
|---|---|---|---|---|
| Sec. 3 | dt-convergence | `runs/dt.sh 1..4` | `src/cmp175.py` | 14 min |
| Sec. 4.1, Fig. 2 | budget window | `runs/window.sh sk=4 ln=0 seed=S s0=X`, `runs/sizeL.sh 72 1` | `src/ana130.py` | — |
| Sec. 4.2, Fig. 3 | trigger, 41 % flat nodes, ≥ 73 nodes, AUC 0.97 | `runs/birth.sh 1..4` | `src/birth111.py`, `bist112.py`, `sel113.py` (env `SNPAT`) | 8 min |
| Sec. 4.3, Fig. 4 | coexistence, P-map, size vs L | `runs/sizeL.sh L S` | `src/ana139.py` | — |
| Sec. 4.4, Fig. 5 | Φ★ only in domains, confinement 42–55 % | `runs/light.sh 1 1` | `src/ph145.py <..._g91.npz> 0.02` | 26 min + 10 min |
| Sec. 4.4 | dispersion ω = ck[1 − (ka)²/96] | — | `src/disp124.py 24` | — |
| Sec. 4.5, Fig. 6 | vortices, planted lifetime 0–14 | `runs/matter.sh 1.5 1`, `runs/matter.sh 1.5 1 100` | `src/ana174.py`, `life174.py`, `wall174.py` | 60 min |
| Sec. 4.6, Fig. 7 | hole seeds a domain: R ≥ 8, q ≥ 0.3; first domain at the hole 3/3 | `runs/hole.sh 48 4.5 SEED R Q` (R = 2–12, Q = 0.05–0.9), `runs/hole.sh 72 3.5 SEED 8 0.6` | `src/ana177.py runs/out/hole_L48` | ≈ 40 min (66 runs) |
| Sec. 4.3, Fig. 4 (Statistics) | window on 7–8 seeds; arrest vs P in v1 seed-dependent (0.3–0.6); merging is not Ostwald ripening | `runs/stats.sh TAG L S0 SEED GAMMA TT` (44 runs, header of the script) | `paper/fig9_stats.py` (data in `data/r178`) | ≈ 75 min on 4 cores |
| Sec. 5.1, Fig. 8a | Nordtvedt effect | — | `src/ff155.py K 0.006`, K = 0, 1, 4, 8 | — |
| Sec. 5.2, Fig. 8b | 3D islands | `src/rg4ck.py` (3D, tetrahedral) | `src/ana170.py` | ≈ 4 h per seed (L = 24) |

## Licence

MIT, see `LICENSE`.
