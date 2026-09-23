"""Melnychuk's spillover-SCM reference: the shipped code, and a converged variant.

Melnychuk (2024), *"Synthetic Controls with Spillover Effects: A Comparative
Study"* (arXiv 2405.01645), ships R implementations of Di Stefano & Mellace's
inclusive SCM and of his own iterative ("waterfall") SCM at
https://github.com/Melnychuk-Andrii/Spillover-SCM. The repo is cloned at a
pinned commit (``_COMMIT``) and run live; the captured run is committed under
``benchmarks/reference/spillsynth_iscm_german/``.

Two reference points are exposed, and they are different numbers:

:func:`shipped_reference_german`
    What the authors' code prints. The no-covariate backend's ``scm_weights``
    reads ``kernlab::ipop``'s primal without checking feasibility, and on this
    panel ipop reports ``how = "converged"`` on a solution whose weights sum to
    0.9666 fitting West Germany and 1.1933 fitting Austria. The reference's
    synthetic controls are therefore not convex combinations, and its inclusive
    estimate (-1651.52) is downstream of that.

:func:`converged_reference_german`
    The same algorithm with the same demeaning, intercept and Cramer's-rule
    correction, with the quadratic program solved to the simplex instead
    (cvxpy/OSQP at 1e-9; SLSQP stalls on this Gram). This is the number the
    reference would print if its solver landed on the constraint set, and it is
    what mlsynth agrees with, to about 3 USD out of 1275.

Keeping both is the point. Validating mlsynth only against the converged variant
would report agreement with a re-implementation instead of with the reference,
and the 376 USD between them would go unrecorded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference import load_reference
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/Melnychuk-Andrii/Spillover-SCM.git"
_COMMIT = "282b621f73f834b0822dcca55d9894c79f335744"
_CACHE = Path(__file__).resolve().parent / ".cache" / "Melnychuk-Andrii-Spillover-SCM"
_BUNDLE = "spillsynth_iscm_german"


def _ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``."""
    marker = _CACHE / "repgermany.dta"
    if marker.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    fetch_pinned_repo(_REPO, _COMMIT, _CACHE)    # git clone, else codeload tarball
    if not marker.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing repgermany.dta")
    return _CACHE


def shipped_reference_german() -> dict:
    """The authors' own R output, from the captured run.

    Returns every quantity ``reference.R`` reports: both backends, both branches
    of ``replacePreTreatData``, the two cross-weights, and the weight sums that
    show the no-covariate solve missing the simplex. Regenerate with
    ``python benchmarks/reference/generate.py spillsynth_iscm_german``.

    The covariate cells come from ``Synth::synth``, whose ``V`` search on this
    specification is sensitive far below any measurement precision: perturbing
    the predictor block by a relative 1e-6 -- the difference between reading
    ``repgermany.dta`` directly and round-tripping it through a CSV -- moves the
    inclusive estimate by over 100 USD. The bundle therefore reads the ``.dta``
    the estimators read, and its provenance records that file's checksum.
    """
    return dict(load_reference(_BUNDLE)["values"])


def _scm_weights(Y_pre: np.ndarray):
    """The reference's ``scm_weights``, solved to the simplex.

    ``Y_pre`` is the pre-period outcome matrix (rows = times, cols = units, col 0
    the target). Returns ``(a, b)`` with intercept ``a`` and full weight vector
    ``b`` (zero on the target). The demeaning, the intercept and the objective
    are the reference's; what differs is that the constraint is enforced --
    cvxpy/OSQP at 1e-9, which is robust on this ill-conditioned outcome Gram
    where SLSQP stalls and where ipop returns an infeasible primal.
    """
    import cvxpy as cp

    means = Y_pre.mean(axis=0)
    D = Y_pre - means
    yt, Xt = D[:, 0], D[:, 1:]
    n1 = Xt.shape[1]
    w = cp.Variable(n1)
    cp.Problem(cp.Minimize(cp.sum_squares(yt - Xt @ w)),
               [cp.sum(w) == 1, w >= 0]).solve(
        solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=100000)
    b = np.zeros(Y_pre.shape[1])
    b[1:] = np.asarray(w.value).ravel()
    a = float(means[0] - means @ b)
    return a, b


def _run_scm(foo: np.ndarray, pre: np.ndarray):
    """Reference ``runSCM``: gap of target (col 0) vs demeaned-intercept SC."""
    a, b = _scm_weights(foo[pre])
    cf = a + foo @ b
    return foo[:, 0] - cf, b


def _german_panel():
    """The repo's own ``repgermany.dta``, wide, treated first, Austria second."""
    import pandas as pd

    d = pd.read_stata(_ensure_clone() / "repgermany.dta")
    piv = d.pivot(index="year", columns="country", values="gdp")
    cols = ["West Germany"] + [c for c in piv.columns if c != "West Germany"]
    piv = piv[cols]
    years = piv.index.to_numpy()
    return piv.to_numpy(), cols, years < 1990, years >= 1990


def converged_reference_german() -> dict:
    """The reference's algorithm on its own data, with the QP solved properly.

    Returns ``{w_A, l_WG, naive_att, inclusive_att, iterative_att,
    iterative_att_replace_pre}`` for West Germany treated with Austria the
    single affected neighbour.
    """
    foo, cols, pre, post = _german_panel()
    aff = cols.index("Austria")

    theta_gap, w_main = _run_scm(foo, pre)          # treated on all donors
    w_A = float(w_main[aff])

    order = list(range(foo.shape[1]))               # swap treated <-> Austria
    order[0], order[aff] = order[aff], order[0]
    gamma_gap, w_a = _run_scm(foo[:, order], pre)   # Austria on all donors
    l_WG = float(w_a[order.index(0)])

    det = 1.0 - w_A * l_WG
    incl_gap = (theta_gap + w_A * gamma_gap) / det

    return {
        "w_A": w_A,
        "l_WG": l_WG,
        "omega_det": det,
        "naive_att": float(theta_gap[post].mean()),
        "inclusive_att": float(incl_gap[post].mean()),
        **_converged_iterative(foo, cols, pre, post, aff),
    }


def _converged_iterative(foo, cols, pre, post, aff) -> dict:
    """Reference ``runIterativeSCM`` (waterfall), both replacePreTreatData branches.

    Austria is swapped into the target slot and the treated unit dropped from
    its donor pool (the reference's ``removeSpUnits``), its synthetic written
    back over the observed series, and West Germany refit on the result.
    """
    out = {}
    order = list(range(foo.shape[1]))
    order[0], order[aff] = order[aff], order[0]
    swapped = foo[:, order]
    keep = [j for j in range(swapped.shape[1]) if j != order.index(0)]
    _, actual_cf, _ = _scm_actual(swapped[:, keep], pre)

    for tag, replace_pre in (("iterative_att", False),
                             ("iterative_att_replace_pre", True)):
        final = foo.copy()
        cleaned = actual_cf.copy()
        if not replace_pre:
            cleaned[pre] = final[pre, aff]          # keep the observed pre-period
        final[:, aff] = cleaned
        gap, _ = _run_scm(final, pre)
        out[tag] = float(gap[post].mean())
    return out


def _scm_actual(foo: np.ndarray, pre: np.ndarray):
    """``runSCM`` returning the fitted path as well as the gap and weights."""
    a, b = _scm_weights(foo[pre])
    cf = a + foo @ b
    return foo[:, 0] - cf, cf, b
