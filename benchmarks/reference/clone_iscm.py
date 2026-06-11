"""On-demand clone of the inclusive-SCM reference repo + a Python port.

Melnychuk (2024), *"Synthetic Controls with Spillover Effects: A Comparative
Study"* (arXiv 2405.01645), ships an R implementation of Di Stefano & Mellace's
**inclusive** SCM at https://github.com/Melnychuk-Andrii/Spillover-SCM. The
reference's synthetic-control backend (``scm_weights``) is a **demeaned simplex
SCM with an intercept**, and the inclusive correction
(``runInclusiveSCM``) solves the cross-weight system by Cramer's rule.

The repo ships the R source and the raw ``repgermany.dta`` panel but no committed
numeric output, and no R runtime is assumed here. We therefore (i) clone the repo
at a pinned commit for provenance + the data, and (ii) transcribe the reference's
no-covariates ``scm_weights`` / ``runInclusiveSCM`` to NumPy and run it on the
repo's own ``repgermany.dta``. mlsynth's ``SPILLSYNTH(method='iscm',
iscm_intercept=True)`` is cross-checked against this independent transcription.

The pinned commit (``_COMMIT``) freezes the reference so the cross-check is
reproducible; bump it deliberately, never silently.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from benchmarks.compare import BenchmarkSkipped

_REPO = "https://github.com/Melnychuk-Andrii/Spillover-SCM.git"
_COMMIT = "282b621f73f834b0822dcca55d9894c79f335744"
_CACHE = Path(__file__).resolve().parent / ".cache" / "Melnychuk-Andrii-Spillover-SCM"


def _ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``."""
    marker = _CACHE / "repgermany.dta"
    if marker.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "-c", "credential.helper=", "clone", "--quiet", _REPO, str(_CACHE)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(_CACHE), "checkout", "--quiet", _COMMIT],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        detail = getattr(exc, "stderr", b"")
        msg = detail.decode(errors="ignore").strip() if detail else str(exc)
        raise BenchmarkSkipped(
            f"could not clone reference repo {_REPO} @ {_COMMIT[:7]}: {msg}"
        ) from exc
    if not marker.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing repgermany.dta")
    return _CACHE


def _scm_weights(Y_pre: np.ndarray):
    """Port of Melnychuk's ``scm_weights``: demeaned simplex SCM + intercept.

    ``Y_pre`` is the pre-period outcome matrix (rows = times, cols = units, col 0
    the target). Returns ``(a, b)`` with intercept ``a`` and full weight vector
    ``b`` (zero on the target). Solved as a simplex-constrained least squares on
    the demeaned series (cvxpy/OSQP -- robust on the ill-conditioned outcome
    Gram, where SLSQP stalls).
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


def reference_inclusive_german() -> dict:
    """Run the ported inclusive SCM on the repo's ``repgermany.dta``.

    Returns the reference ``{w_A, l_WG, naive_att, inclusive_att}`` for West
    Germany (treated) with Austria the single affected neighbour.
    """
    import pandas as pd

    d = pd.read_stata(_ensure_clone() / "repgermany.dta")
    piv = d.pivot(index="year", columns="country", values="gdp")
    cols = ["West Germany"] + [c for c in piv.columns if c != "West Germany"]
    piv = piv[cols]
    years = piv.index.to_numpy()
    pre = years < 1990
    post = ~pre
    foo = piv.to_numpy()
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
        "naive_att": float(theta_gap[post].mean()),
        "inclusive_att": float(incl_gap[post].mean()),
    }
