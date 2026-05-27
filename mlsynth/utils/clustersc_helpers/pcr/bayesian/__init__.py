"""Bayesian PCR-SC weight solver (mlsynth extension; Bayani 2022 Ch. 1).

* :mod:`.posterior` -- core Gaussian-conjugate posterior over the weights
  (:func:`BayesSCM`), relocated from the former shared ``bayesutils``.
* :mod:`.solver` -- :func:`solve_bayesian`, the PCR-SC wrapper that draws from
  that posterior and forms per-period credible bands.
"""

from __future__ import annotations

from .posterior import BayesSCM
from .solver import solve_bayesian

__all__ = ["BayesSCM", "solve_bayesian"]
