"""Cross-validation benchmark: LPCA vs Feng's own R on the Section 5 designs.

Cross-validates the numerical core of mlsynth's ``LPCA`` -- the pseudo-max
distance, the neighbourhood rule, the singular-value rank rule and the local
reconstruction -- against the author's own simulation harness
(``Simulation/Feng-2024_LPCA_simul_suppfuns.R`` in
``yingjieum/Replication_NonlinearFactorModel_2023``) on the three
data-generating processes of Feng (2024) Section 5.

What is being compared, and what is not. Feng's Table 1 is a Monte Carlo: 2000
replications, a fresh panel each time. That design cannot be used to compare two
implementations, because R and NumPy have no shared random stream, so the two
sides would be looking at different data and any agreement could only be
asserted up to Monte Carlo error. Here the panels are fixed instead --
``make_panels.py`` writes them once, both implementations read the same bytes --
which removes the sampling channel and turns every cell below into an exact
claim about the estimators.

The statistical reproduction of Table 1 at the paper's own scale is a separate
exercise and is recorded separately: 500 replications at ``n = p = 1000``,
median disagreement 0.83 Monte Carlo standard errors across the 48 cells, in
``benchmarks/reference/lpca_kansas/`` (``run_simulation.py``, and the README's
Path B section).

The panels are ``n = p = 150`` so they can live under version control. The
neighbourhood grid that implies -- 14, 28, 42 -- covers both branches of the
rank rule, which is inert at 14 and live at 28 and 42; the full-scale grid
(49/99/149) exercises only the live branch.

Statistics, per model and per ``K``, are the ones Table 1 reports: the maximum
absolute error over every cell of the held-out block, and the prediction error
at the three units whose latent variable sits at the 0.1, 0.5 and 0.9 sample
quantiles. Those three have their last-period entry zeroed, which is the
simulation's stand-in for a treated cell.

The reference run also records a global-PCA baseline (``HDRFA::PCA_FN`` for the
factor count, as Feng's script uses). It is reported for context -- Table 1's
comparison is against it -- but not pinned, since it is not mlsynth's code.

Observed agreement. 35 of the 36 cells land within 5e-11, which is the print
precision of the captured reference. The exception is ``model2_K14_mae`` at
1.2e-07, and it is the expected one: mlsynth takes an exact eigendecomposition
of the neighbourhood's Gram matrix where the reference calls ``irlba``, whose
default convergence tolerance is 1e-5 on the relative residual. A maximum over
22 500 cells is the statistic most exposed to that, since it reports whichever
cell the two disagree on most. The 1e-6 tolerance sits an order of magnitude
above it.
"""
from __future__ import annotations

import gzip
import os

import numpy as np

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference import load_reference, reference_value

_HERE = os.path.join(os.path.dirname(__file__), "..", "reference", "lpca_mc")
_CASE = "lpca_mc"
_MODELS = (1, 2, 3)
_MAX_COMPONENTS = 3
_MATCH_FRACTION = 0.5


def _surface(kind: int, alpha: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """The noise-free latent surface, recomputed from the stored latent vectors.

    Kept in step with ``make_panels.py``; the reference script recomputes the
    same thing from the same file via Feng's ``funlist``.
    """
    u, v = alpha[:, None], eta[None, :]
    if kind == 1:
        return (1 / np.sqrt(2 * np.pi) / 0.1) * np.exp(-((u - v) ** 2) / 0.1 ** 2)
    if kind == 2:
        return np.exp(-np.abs(u - v) / 0.1)
    return 1 - (1 + np.exp(15 * (0.8 * np.abs(u - v)) ** 0.8 - 0.1)) ** -1


def _load(kind: int):
    with gzip.open(os.path.join(_HERE, f"panel_model{kind}.csv.gz"), "rt") as fh:
        panel = np.loadtxt(fh, delimiter=",")
    latent = np.loadtxt(os.path.join(_HERE, f"latent_model{kind}.csv"),
                        delimiter=",", skiprows=1)
    evaluated = np.atleast_1d(
        np.loadtxt(os.path.join(_HERE, f"evaluated_model{kind}.csv"), dtype=int)) - 1
    return panel, _surface(kind, latent[:, 0], latent[:, 1]), evaluated


def _k_grid(n: int) -> list[int]:
    """``floor(n^(2/3) * c)`` for ``c`` in 0.5, 1, 1.5 -- the reference's grid.

    Computed the way the R does, in double precision, because that is what puts
    the full-scale grid at 49/99/149 and not 50/100/150.
    """
    K = n ** (2 * 1 / (2 * 1 + 1))
    return [int(np.floor(K * c)) for c in (0.5, 1.0, 1.5)]


def run() -> dict:
    try:
        load_reference(_CASE)
    except FileNotFoundError as exc:
        raise BenchmarkSkipped(str(exc)) from exc

    from mlsynth.utils.lpca_helpers import (
        local_pca_row, neighbour_index, pseudo_max_distance,
    )

    got, rows = {}, []
    for kind in _MODELS:
        panel, truth, evaluated = _load(kind)
        n, p = panel.shape
        match = int(round(p * _MATCH_FRACTION))
        block, target = panel[:, match:], truth[:, match:]
        distance = pseudo_max_distance(panel[:, :match])

        for K in _k_grid(n):
            keep = neighbour_index(distance, K)
            fitted = np.empty_like(target)
            for i in range(n):
                fitted[i], _ = local_pca_row(
                    block, keep[:, i], i, K, n_components=_MAX_COMPONENTS)[:2]
            deviation = np.abs(fitted - target)
            got[f"model{kind}_K{K}_mae"] = float(deviation.max())
            for q, unit in enumerate(evaluated, start=1):
                got[f"model{kind}_K{K}_q{q}"] = float(deviation[unit, -1])
            rows.append({"model": kind, "K": K,
                         "mae": float(deviation.max()),
                         "reference_mae": reference_value(
                             _CASE, f"model{kind}_K{K}_mae")})

    reference = load_reference(_CASE)["values"]
    return {
        **got,
        "rows": rows,
        "gpca_context": {
            f"model{k}": {"mae": reference.get(f"model{k}_gpca_mae"),
                          "rank": reference.get(f"model{k}_gpca_rank")}
            for k in _MODELS
        },
        "mlsynth_call": {"estimator": "LPCA (helpers)",
                         "config": {"max_components": _MAX_COMPONENTS,
                                    "match_fraction": _MATCH_FRACTION}},
        "reference": {"impl": "Feng's simulation harness (live run, captured)",
                      "version": "Replication_NonlinearFactorModel_2023 @ ca34fba"},
    }


def _expected() -> dict:
    """Every LPCA cell the reference recorded, at the tolerance irlba allows."""
    out = {}
    for kind in _MODELS:
        for K in _k_grid(150):
            out[f"model{kind}_K{K}_mae"] = (
                reference_value(_CASE, f"model{kind}_K{K}_mae"), 1e-6)
            for q in (1, 2, 3):
                out[f"model{kind}_K{K}_q{q}"] = (
                    reference_value(_CASE, f"model{kind}_K{K}_q{q}"), 1e-6)
    return out


EXPECTED = _expected()
