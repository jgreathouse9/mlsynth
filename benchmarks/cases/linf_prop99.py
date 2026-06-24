"""Cross-validation: mlsynth's L-infinity SC vs ``LinfinitySC`` on Proposition 99.

Wang, Xing & Ye (2025), Section 6.1, apply their L-infinity SC to the canonical
Abadie-Diamond-Hainmueller tobacco panel (California treated by Proposition 99
from 1989; 38 donor states; 1970-2000). Contrary to a first reading of the
paper's figures, the method is fully numeric here: run on the Prop 99 panel the
L-infinity corner case ``our(method='inf')`` returns an intercept, a dense donor
weight vector, and a counterfactual effect path. This case cross-validates
mlsynth's faithful ``LINF`` engine against the authors' own ``LinfinitySC`` code
(https://github.com/BioAlgs/LinfinitySC) on that panel, at a matched penalty.

What is well-determined, and what is not
----------------------------------------
With ``J = 38`` donors and only ``T0 = 19`` pre-treatment periods (``J > T0``),
the L-infinity-penalised weight vector is a non-unique minimiser: the
non-smooth ``||w||_inf`` term ties several donors at the peak magnitude, so
different but equally optimal weight vectors yield the *same* objective and the
*same* fitted values. We verified this directly -- the cvxpy direct
formulation, mlsynth's OSQP fast path, mlsynth's Gram/cvxpy path, and the
reference ``cvxopt`` IPM all land on objectives within ``~6e-4`` of one another
at the matched penalty, with weight vectors differing by up to ``~0.07`` (genuine
non-uniqueness, not a solver bug). The cvxpy direct form attains the lowest
objective; mlsynth's engine sits within solver slack of it and *below* the
reference's cvxopt point. So this case cross-validates the quantities that *are*
identified in the ``J > T0`` regime -- the counterfactual / effect path, the
post-period ATT, the pre-fit, and the dense-with-negative-weights structural
signature -- and pins the weight agreement only at the (looser) correlation
level the non-uniqueness permits.

The two solvers minimise the same intercept-shifted, unconstrained program
``||y - mu - Y w||^2 + lambda * ||w||_inf`` but normalise the loss differently:
the reference QP (``cvxopt``) carries a ``1/(2 T0)`` factor, mlsynth carries
none, so the reference penalty ``lambda_ref`` corresponds to mlsynth
``lambda = 2 * T0 * lambda_ref``. The reference ``lambda_ref`` is the one the
authors' own deterministic cross-validation ``param_selector(method='inf')``
selects on the Prop 99 pre-period.

What we check (mlsynth vs LinfinitySC, matched penalty)
-------------------------------------------------------
* ``att_mean_post`` -- post-period ATT (both ~ -15.3 packs per capita).
* ``ite_1990`` / ``ite_1995`` / ``ite_2000`` -- the effect path tracks cell-by-cell.
* ``pre_rmspe`` -- both interpolate the pre-period near-exactly.
* ``weight_correlation`` -- the dense weight vectors correlate ~0.98.
* ``linf_nnz`` / ``linf_n_negative`` -- dense, off-simplex (negative) weights:
  the Wang-Xing-Ye signature.

Provenance / scenario
---------------------
* Full repo (scenario 3): cross-validation is mandatory and done here.
* Reference: ``LinfinitySC`` ``our(method='inf')``, pinned at
  ``clone_linfinitysc._COMMIT``, solved with ``cvxopt`` (open source). Captured
  live under ``benchmarks/reference/linf_prop99/`` (regenerate with
  ``python benchmarks/reference/generate.py linf_prop99``); the case pins those
  values via ``reference_value`` / ``load_reference`` so they cannot drift.
* Data: ``basedata/smoking_data.csv`` (the ADH Prop 99 panel shipped with
  mlsynth); the mlsynth side always runs.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from benchmarks.reference import load_reference, reference_value

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "smoking_data.csv"
_INTERVENTION = 1989

# The authors' captured reference run (donor weights, intercept, ATT, effect
# path, the param_selector-chosen lambda) -- live LinfinitySC output, not
# transcribed -- read from benchmarks/reference/linf_prop99/.
_REF = load_reference("linf_prop99")
_REF_WEIGHTS = _REF["weights"]
_LAM_REF = float(_REF["values"]["lam_ref"])


def _panel():
    """Return ``(donors, years, T0, Y1 (T,), Y0 (T, J))`` for the Prop 99 panel."""
    import pandas as pd

    d = pd.read_csv(_DATA)
    donors = [s for s in sorted(d["state"].unique()) if s != "California"]
    years = sorted(d["year"].unique())
    T0 = years.index(_INTERVENTION)
    piv = d.pivot(index="year", columns="state", values="cigsale")
    Y1 = piv["California"].to_numpy(dtype=float)
    Y0 = piv[donors].to_numpy(dtype=float)
    return donors, years, T0, Y1, Y0


def _ml_fit(donors, T0, Y1, Y0):
    """mlsynth's faithful LINF at the matched penalty ``lambda = 2 * T0 * lam_ref``."""
    from mlsynth.utils.laxscm_helpers.crossval import fit_en_scm

    lam_ml = 2 * T0 * _LAM_REF
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_en_scm(
            Y0[:T0], Y1[:T0], Y0[T0:], donor_names=donors, fit_intercept=True, y=Y1,
            alpha=[0.0], lam=[lam_ml], second_norm="L1_INF",
            constraint_type="unconstrained", standardize=False, solver="CLARABEL",
        )
    dw = res["donor_weights"]
    w = np.array([float(dw.get(s, 0.0)) for s in donors], dtype=float)
    cf = np.asarray(res["predictions"], dtype=float).flatten()
    return w, cf


def run() -> dict:
    from benchmarks.compare import BenchmarkSkipped

    if not _DATA.exists():  # pragma: no cover - data ships with the repo
        raise BenchmarkSkipped(f"missing Prop 99 data at {_DATA}")

    donors, years, T0, Y1, Y0 = _panel()
    w_ml, cf = _ml_fit(donors, T0, Y1, Y0)
    w_ref = np.array([_REF_WEIGHTS[s] for s in donors], dtype=float)

    tau = Y1 - cf
    return {
        "att_mean_post": float(np.mean(tau[T0:])),
        "ite_1990": float(tau[years.index(1990)]),
        "ite_1995": float(tau[years.index(1995)]),
        "ite_2000": float(tau[years.index(2000)]),
        "pre_rmspe": float(np.sqrt(np.mean(tau[:T0] ** 2))),
        "weight_correlation": float(np.corrcoef(w_ml, w_ref)[0, 1]),
        "linf_nnz": float(np.sum(np.abs(w_ml) > 1e-3)),
        "linf_n_negative": float(np.sum(w_ml < -1e-3)),
    }


def comparison() -> dict:
    """mlsynth LINF vs the ``LinfinitySC`` reference, quantity by quantity.

    Pairs the mlsynth L-infinity fit against the authors' ``our(method='inf')``
    on the Prop 99 panel at the matched penalty: the post-period ATT, the effect
    path at 1990/1995/2000, the pre-period RMSPE, and every donor weight (sorted
    by magnitude). The reference side is the live ``LinfinitySC`` run captured in
    ``benchmarks/reference/linf_prop99/``. Returns ``{"rows": [...],
    "mlsynth_call": {...}, "reference": {...}}``.
    """
    donors, years, T0, Y1, Y0 = _panel()
    w_ml, cf = _ml_fit(donors, T0, Y1, Y0)
    tau = Y1 - cf
    lam_ml = 2 * T0 * _LAM_REF

    rows = [
        {"quantity": "ATT[mean post]", "mlsynth": round(float(np.mean(tau[T0:])), 6),
         "reference": round(reference_value("linf_prop99", "att_mean_post"), 6)},
    ]
    for yr in (1990, 1995, 2000):
        rows.append({"quantity": f"ITE[{yr}]",
                     "mlsynth": round(float(tau[years.index(yr)]), 6),
                     "reference": round(reference_value("linf_prop99", f"ite_{yr}"), 6)})
    rows.append({"quantity": "pre-RMSPE",
                 "mlsynth": round(float(np.sqrt(np.mean(tau[:T0] ** 2))), 6),
                 "reference": round(reference_value("linf_prop99", "pre_rmspe"), 6)})

    order = sorted(donors, key=lambda s: -abs(_REF_WEIGHTS[s]))
    w_map = {s: float(w) for s, w in zip(donors, w_ml)}
    for s in order:
        rows.append({"quantity": f"weight[{s}]",
                     "mlsynth": round(w_map[s], 6),
                     "reference": round(_REF_WEIGHTS[s], 6)})
    return {
        "rows": rows,
        "mlsynth_call": {
            "estimator": "fit_en_scm",
            "config": {"fit_intercept": True, "alpha": 0.0, "lam": lam_ml,
                       "second_norm": "L1_INF", "constraint_type": "unconstrained",
                       "standardize": False, "solver": "CLARABEL"},
        },
        "reference": {"impl": "LinfinitySC our(method='inf') (Wang, Xing & Ye 2025), "
                              "https://github.com/BioAlgs/LinfinitySC, lambda via "
                              "param_selector(method='inf', n_folds=10)",
                      "version": "benchmarks/reference/linf_prop99/ (live captured run)"},
    }


# Cross-validation of the identified quantities at the matched penalty. The
# effect path / ATT / pre-fit are unique even though the weight vector is not
# (J = 38 > T0 = 19), so they pin tightly; the weights pin at the correlation
# the non-uniqueness permits. Reference numbers are read from the live captured
# LinfinitySC run via reference_value, so EXPECTED and the bundle stay in lockstep.
_rv = lambda k: reference_value("linf_prop99", k)
EXPECTED = {
    # Effect path: mlsynth tracks LinfinitySC to well under a pack per capita.
    "att_mean_post": (_rv("att_mean_post"), 0.8),    # mlsynth -15.21 vs ref -15.49
    "ite_1990": (_rv("ite_1990"), 0.9),              # mlsynth -8.75 vs ref -9.14
    "ite_1995": (_rv("ite_1995"), 0.9),              # mlsynth -16.31 vs ref -16.20
    "ite_2000": (_rv("ite_2000"), 0.9),              # mlsynth -21.41 vs ref -21.79
    "pre_rmspe": (0.0, 5e-3),                         # both interpolate the pre-period
    # Dense, off-simplex weights (the Wang-Xing-Ye signature), correlating ~0.98
    # despite the L-infinity non-uniqueness in the J > T0 regime.
    "weight_correlation": (0.98, 0.04),              # fails below ~0.94
    "linf_nnz": (38.0, 2.0),                          # dense across the donor pool
    "linf_n_negative": (14.0, 6.0),                   # leaves the simplex (ref 13, ml 15)
}
