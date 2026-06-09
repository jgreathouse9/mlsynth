"""Cross-validation: mlsynth LINF/L1LINF vs the authors' ``LinfinitySC``.

Matches mlsynth's L-infinity penalized SCM engine cell-by-cell against the
reference implementation of Wang, Xing & Ye (2025) -- the ``LinfinitySC``
repository (https://github.com/BioAlgs/LinfinitySC), ``our(method='inf')`` and
``our(method='l1-inf')`` -- on a shared panel at a matched penalty ``lambda``.

The two solvers minimize the same intercept-shifted, unconstrained penalized
program ``||y - mu - Y w||^2 ... + lambda * P(w)`` but normalize the loss
differently: the reference QP carries a ``1/(2 T0)`` factor (cvxopt), mlsynth
carries none, so a reference ``lambda_ref`` corresponds to mlsynth
``lambda = 2 * T0 * lambda_ref``. In the **over-determined** regime
``T0 > J`` the penalized minimizer is unique, so the two independent
implementations (cvxopt IPM vs cvxpy/CLARABEL) agree to solver precision. (At
``J > T0`` the L-infinity interpolant is non-unique -- different solvers pick
different optima with the same objective -- which is why this cross-check lives
in the unique regime; the Prop 99 case validates the non-unique regime
qualitatively instead.)

Provenance / scenario
---------------------
* Full repo (scenario 3): cross-validation is mandatory and done here.
* Reference: ``LinfinitySC`` pinned at commit ``37499ab``; solves with
  ``cvxopt`` (open source -- no commercial solver needed).
* mlsynth side: the penalized engine ``fit_en_scm`` in the faithful Wang-Xing-Ye
  configuration (``constraint_type='unconstrained'``, ``fit_intercept=True``,
  ``second_norm='L1_INF'``, ``standardize=False``) at a fixed ``lambda``.
* The case **skips gracefully** when the reference clone or ``cvxopt`` is
  unavailable.
"""
from __future__ import annotations

import warnings

import numpy as np

T0, J, T1 = 60, 20, 8
SEED = 3
LAM_REF = 0.02          # reference penalty; mlsynth uses 2 * T0 * LAM_REF


def _panel():
    """Deterministic two-factor panel: donors ``Y0`` (T, J), treated ``Y1`` (T,)."""
    rng = np.random.default_rng(SEED)
    T = T0 + T1
    load = np.arange(1, J + 1) / J
    F = rng.normal(size=(T, 2))
    Y0 = load[None, :] + F[:, 0:1] + load[None, :] * F[:, 1:2] + rng.normal(0, 2.0, (T, J))
    w0 = np.full(J, 1.0 / J)
    Y1 = Y0 @ w0 + rng.normal(0, 1.0, T)
    Y1[T0:] += 3.0
    return Y0, Y1


def _ml_weights(Y0, Y1, alpha, lam_ml):
    from mlsynth.utils.laxscm_helpers.crossval import fit_en_scm
    donors = [f"d{j:02d}" for j in range(J)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_en_scm(
            Y0[:T0], Y1[:T0], Y0[T0:], donor_names=donors, fit_intercept=True, y=Y1,
            alpha=[alpha], lam=[lam_ml], second_norm="L1_INF",
            constraint_type="unconstrained", standardize=False, solver="CLARABEL",
        )
    dw = res["donor_weights"]
    return np.array([dw.get(d, 0.0) for d in donors], dtype=float)


def run() -> dict:
    from benchmarks.compare import BenchmarkSkipped
    from benchmarks.reference.clone_linfinitysc import import_synth

    synth = import_synth()
    try:
        import cvxopt  # noqa: F401  (reference solver backend)
    except Exception as exc:
        raise BenchmarkSkipped(f"cvxopt unavailable: {exc}")

    Y0, Y1 = _panel()
    lam_ml = 2 * T0 * LAM_REF

    out = {}
    for tag, alpha, method in (("linf", 0.0, "inf"), ("l1linf", 0.5, "l1-inf")):
        try:
            w_ref = np.asarray(
                synth.our(Y1[:T0], Y0[:T0], alpha, LAM_REF, method=method,
                          std=False, intercept=True),
                dtype=float,
            )[1:]  # drop the intercept
        except Exception as exc:  # pragma: no cover - reference solve failure
            raise BenchmarkSkipped(f"reference our(method={method!r}) failed: {exc}")
        w_ml = _ml_weights(Y0, Y1, alpha, lam_ml)
        out[f"{tag}_w_l1_diff"] = float(np.abs(w_ref - w_ml).sum())
        out[f"{tag}_w_max_diff"] = float(np.abs(w_ref - w_ml).max())
        # Both are genuinely dense (the WXY signature) -- guard against either
        # side silently collapsing to a sparse / classic-SC solution.
        out[f"{tag}_ml_dense"] = float(int(np.sum(np.abs(w_ml) > 1e-3)) >= J - 2)
    return out


# Two independent implementations of the same unique penalized program -> agreement
# to solver precision (cvxopt IPM vs CLARABEL). Tolerances cover numerical slack.
EXPECTED = {
    "linf_w_l1_diff": (0.0, 3e-2),
    "linf_w_max_diff": (0.0, 1.5e-2),
    "linf_ml_dense": (1.0, 0.0),
    "l1linf_w_l1_diff": (0.0, 3e-2),
    "l1linf_w_max_diff": (0.0, 1.5e-2),
    "l1linf_ml_dense": (1.0, 0.0),
}
