"""SCTA cross-validation: the paper's stacked design, solved independently.

Sun, Ben-Michael & Feller (2024), *"Temporal Aggregation for the Synthetic
Control Method"* (AEA P&P 114: 614-617) fits one set of donor weights that
jointly balances a high-frequency outcome and its temporal aggregates. Their
Section 2 recipe is four steps: average the pre-period into blocks of ``K``,
stack the block means on top of the disaggregated periods, weight the two halves
by a fixed diagonal :math:`V = \\operatorname{diag}(K\\nu, 1)`, demean each unit
by its pre-period mean, and solve the :math:`V`-weighted simplex least squares.

This case builds that design a second time, straight from the paper's equations
and with no reference to :mod:`mlsynth.utils.scta_helpers`, and solves it with
cvxpy/CLARABEL -- an interior-point conic solver, where mlsynth uses its own
primal active-set QP. Two independent constructions and two independent solvers
then have to produce the same estimator. The pattern is the one
:mod:`benchmarks.cases.linf_crossval_ref` uses.

What the two sides can disagree about
-------------------------------------
The construction (which rows are aggregates, what scaling they carry, what is
demeaned by what) is exact or it is wrong, so any error there shows up as a
gross disagreement, not a tolerance question. The *solve* is the part with a
genuine numerical tolerance, and it runs in one direction: both solvers minimise
the same convex objective over the same simplex, so the smaller objective is the
better solve. Where the two differ here it is CLARABEL that halts above the
optimum, at :math:`\\nu = 4` by 3.2e-6 relative -- the same pattern the
replication page records against ``augsynth``'s ``LowRankQP``, in miniature.

The panel
---------
``basedata/ibex_day_ahead_price.csv``: monthly wholesale day-ahead electricity
prices (EUR/MWh) for 22 European countries, 2015-01 to 2023-12, as redistributed
in ``mharoruiz/ibex`` and used by :mod:`benchmarks.cases.ibex_dap`. Spain is
treated by the June-2022 Iberian exception mechanism. Portugal is treated by the
same mechanism and is dropped, leaving 20 clean donors and 89 pre-treatment
months -- seven whole years plus a five-month tail, the ragged-block case the
paper's own Texas application has (six years plus three months).

This is not the paper's application. The Texas SB8 live-birth panel is not
redistributable here, so the ``augsynth`` cross-validation recorded on
``docs/replications/scta.rst`` cannot be re-run from this repository; what is
durable is this one. The two check different things: the ``augsynth`` comparison
places SCTA against the authors' own library on the authors' data, and this
places SCTA's construction and solve against an independent implementation of
the published recipe, on a panel that ships.

Provenance: Sun, Ben-Michael & Feller (2024) Section 2 (the stacked design and
:math:`V`); Ben-Michael, Feller & Rothstein (2021) Section 4 for the ridge
augmentation (``augsynth``'s ``solve_ridge``); data from OMIE and Fraunhofer ISE
energy-charts (CC BY 4.0) as redistributed in ``mharoruiz/ibex``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "ibex_day_ahead_price.csv")
_TREATED = "ES"
_ALSO_TREATED = "PT"                        # IbEx covers Iberia; not a donor
_TREAT_DATE = pd.Timestamp("2022-06-01")
_K = 12                                     # months per temporal aggregate
_NU_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
_RIDGE_LAMBDAS = (1.0, 10.0)
_HEADLINE_NU = 0.5                          # the paper's equal-weight heuristic


def _panel():
    """Long Spain-treated panel, plus the wide arrays the reference side needs."""
    df = pd.read_csv(os.path.abspath(_DATA), parse_dates=["date"])
    complete = df.groupby("country")["DAP"].transform(lambda s: ~s.isna().any())
    df = df[complete & (df.country != _ALSO_TREATED)].copy()
    df["treat"] = ((df.country == _TREATED) & (df.date >= _TREAT_DATE)).astype(int)

    wide = df.pivot(index="date", columns="country", values="DAP").sort_index()
    T0 = int((wide.index < _TREAT_DATE).sum())
    donors = [c for c in wide.columns if c != _TREATED]
    return (df, wide[_TREATED].to_numpy(float), wide[donors].to_numpy(float),
            donors, T0)


# --------------------------------------------------------------------------
# The reference side: the paper's Section 2 recipe, written from the equations.
# --------------------------------------------------------------------------

def _design(y, Yd, T0, nu):
    """``(sqrt(V) a, sqrt(V) B)`` for the stacked design, plus the demeaning.

    ``a`` is the treated unit's matching vector: ``T0 / K`` block means on top of
    the ``T0`` disaggregated pre-periods, each demeaned by that unit's own
    pre-period average. ``B`` is the same for the donors. ``sqrt(V)`` scales the
    aggregate rows by ``sqrt(K * nu)`` so an ordinary least-squares solve on the
    scaled matrices is the ``V``-weighted one.
    """
    J, n_blocks = Yd.shape[1], T0 // _K
    mu_t = y[:T0].mean()
    mu_d = Yd[:T0].mean(axis=0)
    a_dis, B_dis = y[:T0] - mu_t, Yd[:T0] - mu_d
    head = n_blocks * _K                                  # the ragged tail stays
    a_agg = a_dis[:head].reshape(n_blocks, _K).mean(axis=1)
    B_agg = B_dis[:head].reshape(n_blocks, _K, J).mean(axis=1)
    root = np.sqrt(np.concatenate([np.full(n_blocks, _K * nu), np.ones(T0)]))
    return (root * np.concatenate([a_agg, a_dis]),
            root[:, None] * np.vstack([B_agg, B_dis]), mu_t, mu_d)


def _clarabel_simplex(A, B):
    """``argmin ||A - B g||^2`` over the simplex, by interior point."""
    import cvxpy as cp

    g = cp.Variable(B.shape[1])
    cp.Problem(cp.Minimize(cp.sum_squares(A - B @ g)),
               [g >= 0, cp.sum(g) == 1]).solve(
        solver=cp.CLARABEL, tol_gap_abs=1e-13, tol_gap_rel=1e-13, tol_feas=1e-13)
    w = np.clip(np.asarray(g.value, dtype=float).ravel(), 0.0, None)
    return w / w.sum()


def _reference_fit(y, Yd, T0, nu, ridge_lambda=None):
    """Reference weights, counterfactual and ATT at one ``nu``.

    With ``ridge_lambda`` the augmented estimator of Ben-Michael, Feller &
    Rothstein is applied on top: center each matching row by its donor mean,
    take the simplex fit there, and add the ridge correction
    ``(A - B w) (B B' + lambda I)^{-1} B`` that closes the residual imbalance.
    """
    A, B, mu_t, mu_d = _design(y, Yd, T0, nu)
    if ridge_lambda is None:
        w = _clarabel_simplex(A, B)
    else:
        center = B.mean(axis=1)                       # augsynth's X_cent
        Ac, Bc = A - center, B - center[:, None]
        w0 = _clarabel_simplex(Ac, Bc)
        X = np.linalg.solve(Bc @ Bc.T + ridge_lambda * np.eye(Bc.shape[0]), Bc)
        w = w0 + (Ac - Bc @ w0) @ X
    cf = mu_t + (Yd - mu_d) @ w
    return w, float(np.mean((y - cf)[T0:]))


def _objective(y, Yd, T0, nu, w):
    """The V-weighted pre-treatment loss the two solvers both minimise."""
    A, B, _, _ = _design(y, Yd, T0, nu)
    return float(np.sum((A - B @ w) ** 2))


# --------------------------------------------------------------------------
# The mlsynth side
# --------------------------------------------------------------------------

def _scta(df, **kw):
    from mlsynth import SCTA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SCTA({"df": df, "outcome": "DAP", "treat": "treat",
                     "unitid": "country", "time": "date", "block_length": _K,
                     "display_graphs": False, **kw}).fit()


def _weight_vector(res, donors):
    """mlsynth's donor weights as an array aligned to ``donors``."""
    w = res.weights.donor_weights
    return np.array([w.get(d, 0.0) for d in donors], dtype=float)


def _both_sides():
    """Every quantity both implementations produce, paired.

    One SCTA call carries the whole ``nu`` sweep -- the frontier reports each
    grid point's ATT and both imbalance RMSEs -- so the plain comparison costs a
    single fit. The ridge fits are separate calls because augmentation is not a
    frontier axis.
    """
    df, y, Yd, donors, T0 = _panel()
    res = _scta(df, nu=_HEADLINE_NU, frontier=list(_NU_GRID))
    frontier = {round(float(p["nu"]), 4): p for p in res.frontier}

    plain = []
    for nu in _NU_GRID:
        point = frontier[round(float(nu), 4)]
        w_ref, att_ref = _reference_fit(y, Yd, T0, nu)
        plain.append({"nu": nu, "att_ml": float(point["att"]), "att_ref": att_ref,
                      "rmse_dis": float(point["rmse_dis"]),
                      "rmse_agg": float(point["rmse_agg"]),
                      "obj_ref": _objective(y, Yd, T0, nu, w_ref)})
    # Weights come off the headline fit, the only nu whose donor weights the
    # result object exposes; the frontier reports imbalances and ATTs, not w.
    w_head = _weight_vector(res, donors)
    w_ref_head, _ = _reference_fit(y, Yd, T0, _HEADLINE_NU)

    ridge = []
    for lam in _RIDGE_LAMBDAS:
        r = _scta(df, nu=_HEADLINE_NU, augment="ridge", ridge_lambda=lam)
        _, att_ref = _reference_fit(y, Yd, T0, _HEADLINE_NU, ridge_lambda=lam)
        ridge.append({"lambda": lam, "att_ml": float(r.effects.att),
                      "att_ref": att_ref})
    return {"res": res, "plain": plain, "ridge": ridge, "donors": donors,
            "T0": T0, "y": y, "Yd": Yd,
            "w_head": w_head, "w_ref_head": w_ref_head}


def run() -> dict:
    b = _both_sides()
    res, plain = b["res"], b["plain"]

    # The solve, in the one direction it can differ: mlsynth's objective minus
    # the reference's, relative. Negative means mlsynth found the lower value.
    # Only the headline nu, since it is the one fit whose weights the result
    # object exposes.
    head = next(r for r in plain if r["nu"] == _HEADLINE_NU)
    obj_ml = _objective(b["y"], b["Yd"], b["T0"], _HEADLINE_NU, b["w_head"])
    excess = (obj_ml - head["obj_ref"]) / head["obj_ref"]

    rmse_agg = [row["rmse_agg"] for row in plain]
    rmse_dis = [row["rmse_dis"] for row in plain]
    return {
        # the design the paper prescribes, as built
        "n_donors": float(len(b["donors"])),
        "pre_periods": float(b["T0"]),
        "n_blocks": float(res.method_details.parameters_used["n_blocks"]),
        # construction + solve, against the independent implementation
        "max_att_diff_vs_reference": max(abs(r["att_ml"] - r["att_ref"])
                                         for r in plain),
        "headline_weight_l1_vs_reference": float(
            np.abs(b["w_head"] - b["w_ref_head"]).sum()),
        "headline_objective_excess": float(excess),
        "max_ridge_att_diff_vs_reference": max(abs(r["att_ml"] - r["att_ref"])
                                               for r in b["ridge"]),
        # the frontier the paper's Figure 1 draws: aggregation buys aggregate
        # balance and spends disaggregated balance, monotonically in nu
        "rmse_agg_falls_in_nu": float(all(np.diff(rmse_agg) < 0)),
        "rmse_dis_rises_in_nu": float(all(np.diff(rmse_dis) > 0)),
        "rmse_agg_at_nu_zero": rmse_agg[0],
        "rmse_agg_at_nu_four": rmse_agg[-1],
        "rmse_dis_at_nu_zero": rmse_dis[0],
        "rmse_dis_at_nu_four": rmse_dis[-1],
        # the estimate itself, and how little the aggregation knob moves it here
        "att_at_headline_nu": float(res.effects.att),
        "att_range_over_nu": max(r["att_ml"] for r in plain)
        - min(r["att_ml"] for r in plain),
    }


def comparison() -> dict:
    """SCTA against the independent construction, cell by cell."""
    b = _both_sides()
    rows = [{"quantity": f"ATT (nu={r['nu']:g})",
             "mlsynth": round(r["att_ml"], 6),
             "reference": round(r["att_ref"], 6)} for r in b["plain"]]
    rows.append({"quantity": f"donor-weight L1 (nu={_HEADLINE_NU:g})",
                 "mlsynth": round(float(np.abs(b["w_head"]
                                               - b["w_ref_head"]).sum()), 8),
                 "reference": 0.0})
    rows += [{"quantity": f"ridge ATT (nu={_HEADLINE_NU:g}, lambda={r['lambda']:g})",
              "mlsynth": round(r["att_ml"], 6),
              "reference": round(r["att_ref"], 6)} for r in b["ridge"]]
    return {
        "rows": rows,
        "mlsynth_call": {
            "estimator": "SCTA",
            "config": {"block_length": _K, "nu": list(_NU_GRID),
                       "augment": [None, "ridge"],
                       "ridge_lambda": list(_RIDGE_LAMBDAS),
                       "panel": "ibex day-ahead price, ES treated 2022-06"}},
        "reference": {
            "impl": "independent implementation of Sun, Ben-Michael & Feller "
                    "(2024) Sec. 2 stacked design, solved by cvxpy/CLARABEL",
            "version": f"cvxpy CLARABEL interior point, tol 1e-13; "
                       f"ridge per Ben-Michael-Feller-Rothstein solve_ridge"},
    }


# Tolerances. The construction quantities (donor count, pre-periods, blocks) are
# counts and carry none. The agreement quantities are all differences that ought
# to be zero, so the tolerance is the numerical floor of each comparison:
#
#   ATT              1e-7   -- both sides evaluate the same closed form on the
#                             same arrays once the weights agree; observed 2e-11
#                             plain and 4e-10 ridge, so this is three orders of
#                             headroom for a different CLARABEL build.
#   weight L1        1e-4   -- floored by mlsynth, not by either solver:
#                             donor_weights is rounded to six decimals, so 20
#                             donors cannot agree closer than 1e-5. Observed 1e-6.
#   objective excess 1e-6   -- signed, and the informative direction is that it
#                             is not positive by more than solver noise: a
#                             positive excess would mean mlsynth had stopped
#                             above the reference optimum. Observed 1.4e-7.
#
# The frontier and estimate quantities are the panel's own values and carry
# bands wide enough to absorb a solver change but tight enough to catch a
# construction change: the imbalance RMSEs to 1e-2 (about 0.5 percent of their
# magnitude) and the ATT to 1e-3 EUR/MWh on an effect of -44.
EXPECTED = {
    "n_donors": (20.0, 0.0),
    "pre_periods": (89.0, 0.0),
    "n_blocks": (7.0, 0.0),
    "max_att_diff_vs_reference": (0.0, 1e-7),
    "headline_weight_l1_vs_reference": (0.0, 1e-4),
    "headline_objective_excess": (1.36e-7, 1e-6),
    "max_ridge_att_diff_vs_reference": (0.0, 1e-7),
    "rmse_agg_falls_in_nu": (1.0, 0.0),
    "rmse_dis_rises_in_nu": (1.0, 0.0),
    "rmse_agg_at_nu_zero": (2.3662, 1e-2),
    "rmse_agg_at_nu_four": (1.8236, 1e-2),
    "rmse_dis_at_nu_zero": (6.4633, 1e-2),
    "rmse_dis_at_nu_four": (6.7021, 1e-2),
    "att_at_headline_nu": (-44.3238, 1e-3),
    "att_range_over_nu": (0.3915, 1e-3),
}
