# Variant code (paper Sec. 4.7)

A snapshot of the model code used for the variant of Sec. 4.7: P = 0.16 (ℓ = 8, γ = 0.0025), cubic term b = 0.1 (`pb=0.1`),
β = 1.3, random-field arrows with scale ℓ_τ = 16 (`ell=16 iso=1`). The entry point is `run268.py` (run it, do not import it:
it parses its arguments and starts the simulation at module level). See `runs/variant.sh`.

The modules are the same family as those in `src/` for model v1, extended with the cubic term, frequent vortex frames (`fine=`)
and checkpoint/resume. Comments in this directory and in `src/hol284.py`, `abw285.py`, `realw286.py`, `ana285.py`, `ana286.py`
are in Ukrainian; the code itself is unchanged from the runs reported in the paper.
