"""Dump the exact donor/treated matrices the benchmark's solver sees, per cell."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
from mlsynth.utils.dsc_helpers.simulate import (
    draw_model_free_design, simulate_model_free_period,
    model_free_risk_matrices, population_risk, oracle_simplex_weights)

OUT = Path(__file__).resolve().parent
SEED0 = 20_240_953
CELLS = [(50, 50), (50, 100), (20, 50), (20, 400)]   # degenerate -> well-conditioned

for J, M in CELLS:
    rng = np.random.default_rng(SEED0 + 1000 * J + M)     # replication r = 0
    mu, sigma = draw_model_free_design(J, rng)
    donors, treated = simulate_model_free_period(mu, sigma, M, rng)
    np.savetxt(OUT / f"cell_J{J}_M{M}_A.csv", donors, delimiter=",")
    np.savetxt(OUT / f"cell_J{J}_M{M}_b.csv", treated, delimiter=",")
    np.savetxt(OUT / f"cell_J{J}_M{M}_design.csv",
               np.column_stack([mu, sigma]), delimiter=",", header="mu,sigma")
    G, bb, c = model_free_risk_matrices(mu, sigma)
    np.savetxt(OUT / f"cell_J{J}_M{M}_oracle.csv", oracle_simplex_weights(G, bb), delimiter=",")
    print(f"J={J} M={M}: A{donors.shape} cond={np.linalg.cond(donors):.3e}")
