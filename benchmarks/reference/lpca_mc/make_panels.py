"""Write the fixed simulation panels the ``lpca_mc`` cross-check runs on.

    python benchmarks/reference/lpca_mc/make_panels.py

Feng's Section 5 draws a fresh panel per replication, which is the right design
for a Monte Carlo but the wrong one for a cross-implementation check: R and
NumPy cannot be made to produce the same random stream, so any comparison would
be statistical and could only ever bound the difference by Monte Carlo error.
Fixing the panels instead removes the sampling channel entirely. Both
implementations read the same bytes, so the two estimators can be compared cell
by cell and any disagreement is theirs.

The three data-generating processes are the paper's, at ``n = p = 150`` in place
of 1000 so the panels are small enough to keep under version control. The
neighbourhood grid that size implies -- 14, 28, 42 -- is a useful accident: the
singular-value ratio rule is inert at 14 and live at 28 and 42, so one run
exercises both branches.

The Table 1 reproduction at the paper's own scale is a separate, statistical
exercise; it lives in ``benchmarks/reference/lpca_kansas/run_simulation.py``.
"""

from __future__ import annotations

import gzip
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N = P = 150
SEED = 20240731
MODELS = (1, 2, 3)


def surface(kind: int, alpha: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """The latent surface (R: ``hdfun1`` / ``hdfun2`` / ``hdfun3``).

    ``sig`` is fixed at 0.1 in the reference code, so model 1's exponent is
    ``-(u - v)^2 / 0.01``; the paper's Section 5 prints ``-10(u - v)^2``. The
    code is what Table 1 was produced from.
    """
    u, v = alpha[:, None], eta[None, :]
    if kind == 1:
        return (1 / np.sqrt(2 * np.pi) / 0.1) * np.exp(-((u - v) ** 2) / 0.1 ** 2)
    if kind == 2:
        return np.exp(-np.abs(u - v) / 0.1)
    return 1 - (1 + np.exp(15 * (0.8 * np.abs(u - v)) ** 0.8 - 0.1)) ** -1


def build(kind: int, rng: np.random.Generator):
    """One panel: the observed matrix, the noise-free truth, the evaluated units.

    Mirrors R's ``dgp`` followed by ``sim``'s ``x[eval.ind, p] <- 0`` -- the
    three units at the 0.1, 0.5 and 0.9 quantiles of the latent variable have
    their last-period entry zeroed, standing in for a treated cell.
    """
    sigma, binary = (1.0, True) if kind == 3 else (0.5, False)
    alpha = rng.uniform(0, 1, N)
    eta = rng.uniform(0, 1, P)
    truth = surface(kind, alpha, eta)
    if binary:
        x = (rng.uniform(0, 1, (N, P)) <= truth).astype(float)
    else:
        x = truth + rng.normal(0, sigma, (N, P))
    order = np.argsort(alpha, kind="stable")
    evaluated = order[[int(round(q * N)) - 1 for q in (0.1, 0.5, 0.9)]]
    x[evaluated, P - 1] = 0.0
    return x, alpha, eta, evaluated


def main() -> int:
    seeds = np.random.SeedSequence(SEED).spawn(len(MODELS))
    for kind, seed in zip(MODELS, seeds):
        x, alpha, eta, evaluated = build(kind, np.random.default_rng(seed))
        with gzip.open(os.path.join(HERE, f"panel_model{kind}.csv.gz"),
                       "wt", compresslevel=9) as fh:
            np.savetxt(fh, x, fmt="%.12g", delimiter=",")
        # The noise-free truth is a closed form of the two latent vectors, so
        # both implementations recompute it from these 300 numbers instead of
        # carrying a second matrix.
        np.savetxt(os.path.join(HERE, f"latent_model{kind}.csv"),
                   np.column_stack([alpha, eta]), fmt="%.17g",
                   delimiter=",", header="alpha,eta", comments="")
        # 1-based for R; the Python side subtracts one.
        np.savetxt(os.path.join(HERE, f"evaluated_model{kind}.csv"),
                   evaluated + 1, fmt="%d")
        print(f"model {kind}: {x.shape} panel, evaluated units "
              f"{(evaluated + 1).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
