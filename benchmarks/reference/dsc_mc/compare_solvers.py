"""mlsynth's simplex solver against DiSCo's, on bit-identical matrices."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
from mlsynth.utils.dsc_helpers.weights import solve_simplex_weights
from mlsynth.utils.dsc_helpers.simulate import (
    model_free_risk_matrices, population_risk)

HERE = Path(__file__).resolve().parent
CELLS = [(50, 50), (50, 100), (20, 50), (20, 400)]
print(f"{'cell':>12} {'max|dw|':>10} {'sum|dw|':>10} "
      f"{'obj mlsynth':>14} {'obj DiSCo':>14} {'rel obj gap':>12} "
      f"{'risk mlsynth':>13} {'risk DiSCo':>12}")
for J, M in CELLS:
    tag = f"J{J}_M{M}"
    A = np.loadtxt(HERE / f"cell_{tag}_A.csv", delimiter=",")
    b = np.loadtxt(HERE / f"cell_{tag}_b.csv", delimiter=",")
    des = np.loadtxt(HERE / f"cell_{tag}_design.csv", delimiter=",", skiprows=1)
    mu, sigma = des[:, 0], des[:, 1]
    G, bb, c = model_free_risk_matrices(mu, sigma)

    w_ml = solve_simplex_weights(A, b)
    w_r = np.atleast_1d(np.loadtxt(HERE / f"cell_{tag}_w_disco.csv"))
    obj = lambda w: float(np.sum((A @ w - b) ** 2))
    o_ml, o_r = obj(w_ml), obj(w_r)
    print(f"{tag:>12} {np.max(np.abs(w_ml - w_r)):10.3e} "
          f"{np.sum(np.abs(w_ml - w_r)):10.3e} {o_ml:14.8f} {o_r:14.8f} "
          f"{abs(o_ml - o_r) / max(o_r, 1e-12):12.3e} "
          f"{population_risk(w_ml, G, bb, c):13.6f} "
          f"{population_risk(w_r, G, bb, c):12.6f}")
