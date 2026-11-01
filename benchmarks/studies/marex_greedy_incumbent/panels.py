"""Five panels of 18 markets, as MAREX's exact backend sees one cluster.

Each returns ``(B, A)``: the markets as the columns of ``B`` over the fit
window, and the target ``A``, their mean (the base design with ``f_j = 1/J``).

- ``section5``: Abadie and Zhao's Section 5 data-generating process with
  ``J = 18`` markets in place of 15, drawn with NumPy (not the authors' R
  seeds): ``T = 30``, fit on the first ``T_E = 20`` periods, ``R = 7``
  observed and ``F = 11`` unobserved covariates. ``delta_t`` is a sorted draw of
  Uniform(0, 20), ``theta_t`` and ``lambda_t`` are Uniform(0, 10), ``Z_j`` and
  ``mu_j`` are Uniform(0, 1), the noise is N(0, 1). The design matches on the
  fit-period outcomes, as ``benchmarks/cases/marex_section5_mc.py`` configures
  MAREX.
- ``geolift``: 18 of GeoLift's 40 markets, drawn once, over the first 90 days.
- ``walmart``: 18 of the 45 stores of the Walmart panel as the exact backend
  sees it in the paper's design (standardised outcomes and covariates,
  ``benchmarks/studies/marex_bnb/results/walmart_panel.npz``), drawn once; the
  target is the 18 stores' mean.
- ``noise`` and ``factor``: the ``marex_bnb`` study's synthetic panels at 18
  markets and 40 periods; ``factor`` has four markets that track the mean.
"""
from __future__ import annotations

import numpy as np

from benchmarks.studies.marex_bnb import panels as bnb

#: Markets per panel.
J = 18


def _subset(B: np.ndarray, seed: int):
    cols = np.sort(np.random.default_rng(seed).choice(B.shape[1], J, replace=False))
    B = B[:, cols]
    return B, B.mean(axis=1)


def section5(seed: int = 1, T: int = 30, TE: int = 20, R: int = 7, F: int = 11):
    """The Section 5 process at 18 markets; the fit window's outcomes."""
    rng = np.random.default_rng(seed)
    delta = np.sort(rng.uniform(0.0, 20.0, T))
    theta = rng.uniform(0.0, 10.0, (T, R))
    lam = rng.uniform(0.0, 10.0, (T, F))
    Z = rng.uniform(0.0, 1.0, (R, J))
    mu = rng.uniform(0.0, 1.0, (F, J))
    Y = delta[:, None] + theta @ Z + lam @ mu + rng.normal(0.0, 1.0, (T, J))
    B = Y[:TE]
    return B, B.mean(axis=1)


def geolift(seed: int = 0):
    """18 of GeoLift's markets over the first 90 days."""
    return _subset(bnb.geolift(90)[0], seed)


def walmart(seed: int = 0):
    """18 of the Walmart stores, as the exact backend sees them."""
    return _subset(bnb.walmart()[0], seed)


def noise():
    """A common shock and no structure."""
    return bnb.noise(J, 40, seed=3)


def factor():
    """Two factors; four markets track the mean loading with little noise."""
    return bnb.factor(J, 40, seed=1)


PANELS = {"section5": section5, "geolift": geolift, "walmart": walmart,
          "noise": noise, "factor": factor}
