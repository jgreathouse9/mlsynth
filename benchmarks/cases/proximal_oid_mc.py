"""PIOID Path-B: the authors' linear interactive-fixed-effects Monte Carlo.

Validates mlsynth's over-identified proximal-inference estimator (``PIOID``)
against the linear simulation the authors ship in their manuscript replication
(``KenLi93/proximal_sc_manuscript``, ``simulation/run_sim_linear_est.R``),
accompanying

    Shi, X., Li, K. T., Yu, Y., Miao, W., Kuchibhotla, A. K., Hu, W., &
    Tchetgen Tchetgen, E. J. (2026). "Theory for Identification and Inference
    with Synthetic Controls: A Proximal Causal Inference Framework." JASA.

The DGP (:func:`mlsynth.utils.proximal_helpers.simulation.simulate_pioid_linear`)
is a linear factor model ``Y_it = U_i' lambda_t + eps_it`` with ``true.beta = 2``:
one treated unit and ``n_units - 1`` controls, split into treatment proxies ``Z``
and donor outcomes ``W`` that share the (diagonal) loading structure but carry
*independent* idiosyncratic noise. That shared error-in-variables structure is
the point: a naive synthetic control regresses the treated series on the noisy
donors ``W`` and inherits an attenuation-style bias, while PIOID instruments
``W`` with ``Z`` and stays consistent.

The case asserts the two headline facts of the manuscript's linear table:

1. Recovery + honest inference. PIOID recovers ``true.beta = 2`` (mean ATT ~ 2.0,
   bias ~ 0) with over-identified GMM/Newey-West Wald coverage near the nominal
   95% (mildly anticonservative at the small ``n_units = 7`` grid point, ~0.93).
2. Proximal beats naive SC. A simplex-constrained synthetic control fit to the
   same donors ``W`` carries a sizeable positive bias (~ +0.54), and PIOID is
   strictly less biased.

  =====================  ===========  ===============
  Quantity               mlsynth      reference
  =====================  ===========  ===============
  PIOID mean ATT         ~2.0         2.0 (true.beta)
  PIOID 95% coverage     ~0.93        ~0.95 (nominal)
  naive SC bias          ~ +0.54      biased (EIV)
  PIOID less biased      True         yes
  =====================  ===========  ===============

Path B (the authors' own DGP): the case asserts the geometry -- PIOID unbiased
with near-nominal coverage and strictly less biased than the naive SC it is
meant to correct -- not exact Monte Carlo cells.

Two mutually reinforcing pieces, both cheap:

* Statistical core (``M = 15000`` reps). Driven at the array level
  (:func:`..pi.overid.estimate_pi_overid` for the over-identified 2SLS bridge,
  :func:`..bilevel.active_set.solve_simplex_qp` for the simplex SC) at ~0.2 ms
  a rep, so 15k reps run in ~3.5 s. The large ``M`` makes the pinned centres
  precise (the Monte Carlo standard error on the mean bias is ~0.003), so the
  tolerances are tight.
* Full-``.fit()`` equivalence guard (``N_FIT`` reps). The array path is a
  shortcut *only* for speed: ``PROXIMAL(...).fit()`` builds a DataFrame, runs
  ``dataprep`` and calls the very same ``estimate_pi_overid`` internals. This
  block runs the public API on a handful of independent draws and asserts the
  ATT and its SE match the array path to solver tolerance (~1e-9), so the
  15k-rep shortcut can never silently drift from what a user's ``.fit()`` call
  returns.

The default grid is the paper's ``n_units = 7``, ``t0 = 80`` (so ``T = 160``,
equal pre/post), unconstrained treated loading, stationary factors, i.i.d.
errors at ``mysd = 1.5``. The GMM HAC lag is the manuscript's Newey-West
``q = 10``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mlsynth.utils.proximal_helpers.simulation import simulate_pioid_linear

_TRUE = 2.0
_HAC = 10          # manuscript's Newey-West lag
M = 15000          # array-level recovery / coverage replications
N_FIT = 25         # full-.fit() equivalence-guard replications


def _draw(rng):
    return simulate_pioid_linear(
        n_units=7, t0=80, true_att=_TRUE, u_setting="unconstrained",
        dist_lambda="stationary", dist_epsilon="iid", rng=rng)


def _mc(base_seed: int):
    """Mean ATT + 95% coverage of PIOID, and the naive-SC bias, over ``M`` draws."""
    from mlsynth.utils.proximal_helpers.pi.overid import estimate_pi_overid
    from mlsynth.utils.bilevel.active_set import solve_simplex_qp

    rng = np.random.default_rng(base_seed)
    pioid_att = np.empty(M)
    pioid_cov = np.empty(M)
    sc_att = np.empty(M)
    for i in range(M):
        s = _draw(rng)
        T = 2 * s.T0
        n_post = T - s.T0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cf, _alpha, se = estimate_pi_overid(
                s.y, s.donor_outcomes, s.donor_proxies, s.T0, n_post, T, _HAC)
            # Naive SCM: simplex-constrained fit of the treated series on the
            # noisy donors W (min ||W w - y||^2 on the pre-period, w on the
            # simplex) -- the estimator PIOID is meant to de-bias.
            Wp, Yp = s.donor_outcomes[:s.T0], s.y[:s.T0]
            w_sc = solve_simplex_qp(Wp, Yp)
        tau = float(np.mean((s.y - cf)[s.T0:]))
        pioid_att[i] = tau
        pioid_cov[i] = abs(tau - _TRUE) <= 1.96 * se if np.isfinite(se) else np.nan
        sc_att[i] = float(np.mean((s.y - s.donor_outcomes @ w_sc)[s.T0:]))
    return (float(pioid_att.mean()), float(np.nanmean(pioid_cov)),
            float(sc_att.mean()))


def _draw_to_frame(s):
    """Long-format panel for the treated unit + W (donors) + Z (instruments)."""
    T = 2 * s.T0
    W, Z, y = s.donor_outcomes, s.donor_proxies, s.y
    rows = [{"unit": "treated", "t": k, "y": float(y[k]), "treat": int(k >= s.T0)}
            for k in range(T)]
    for j in range(Z.shape[1]):
        rows += [{"unit": f"Z{j}", "t": k, "y": float(Z[k, j]), "treat": 0}
                 for k in range(T)]
    for j in range(W.shape[1]):
        rows += [{"unit": f"W{j}", "t": k, "y": float(W[k, j]), "treat": 0}
                 for k in range(T)]
    donors = [f"W{j}" for j in range(W.shape[1])]
    instruments = [f"Z{j}" for j in range(Z.shape[1])]
    return pd.DataFrame(rows), donors, instruments


def _fit_vs_array(base_seed: int):
    """Max |ATT| and |SE| gap between PROXIMAL(...).fit() and the array path.

    Runs the public ``PROXIMAL`` API -- DataFrame, ``dataprep``, results object
    -- and the direct ``estimate_pi_overid`` call on the *same* ``N_FIT`` draws,
    and returns the largest absolute discrepancy in the ATT and its SE. Both are
    the same estimator, so the gap is solver round-off (~1e-15); the guard pins
    it well below any meaningful drift.
    """
    from mlsynth import PROXIMAL
    from mlsynth.utils.proximal_helpers.pi.overid import estimate_pi_overid

    rng = np.random.default_rng(base_seed)
    att_gap = 0.0
    se_gap = 0.0
    for _ in range(N_FIT):
        s = _draw(rng)
        T = 2 * s.T0
        n_post = T - s.T0
        df, donors, instruments = _draw_to_frame(s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = PROXIMAL({
                "df": df, "outcome": "y", "treat": "treat",
                "unitid": "unit", "time": "t",
                "donors": donors, "outcome_instruments": instruments,
                "methods": ["PIOID"], "display_graphs": False,
            }).fit().methods["PIOID"]
            cf, _alpha, arr_se = estimate_pi_overid(
                s.y, s.donor_outcomes, s.donor_proxies, s.T0, n_post, T, _HAC)
        arr_att = float(np.mean((s.y - cf)[s.T0:]))
        fit_se = float(fit.att_se) if fit.att_se is not None else float("nan")
        att_gap = max(att_gap, abs(float(fit.att) - arr_att))
        se_gap = max(se_gap, abs(fit_se - arr_se))
    return att_gap, se_gap


def run() -> dict:
    pioid_mean, pioid_cov, sc_mean = _mc(2026)
    pioid_bias = pioid_mean - _TRUE
    sc_bias = sc_mean - _TRUE
    fit_att_gap, fit_se_gap = _fit_vs_array(909)
    return {
        "pioid_att": pioid_mean,
        "pioid_bias": pioid_bias,
        "pioid_coverage": pioid_cov,
        "naive_sc_att": sc_mean,
        "naive_sc_bias": sc_bias,
        "pioid_less_biased_than_sc": float(abs(pioid_bias) < abs(sc_bias)),
        # Full-.fit() equivalence guard: the public API matches the array path.
        "fit_vs_array_att_max_diff": fit_att_gap,
        "fit_vs_array_se_max_diff": fit_se_gap,
    }


# Deterministic (array core seeded at base 2026, M=15000; equivalence guard at
# base 909, N_FIT=25). At M=15000 the Monte Carlo standard error on the mean bias
# is ~0.003 and the cross-seed spread is ~+/-0.01 on every quantity (PIOID bias in
# [-0.03, -0.01], coverage in [0.932, 0.938], naive-SC bias in [+0.536, +0.541]),
# so the tolerances are tight. The reproduced facts: PIOID recovers true.beta=2
# (small finite-sample bias ~ -0.02) with near-nominal, mildly anticonservative
# over-identified GMM coverage, and is strictly less biased than the simplex SC
# fit to the same error-prone donors. The equivalence guard pins PROXIMAL(...).fit()
# to the array path at solver round-off, so the 15k-rep shortcut cannot drift from
# the public API.
EXPECTED = {
    "pioid_bias": (0.0, 0.06),                  # PIOID recovers true.beta=2
    "pioid_coverage": (0.93, 0.03),             # near nominal 0.95 (mildly anticonservative)
    "naive_sc_bias": (0.54, 0.05),              # naive SC biased by EIV in the donors
    "pioid_less_biased_than_sc": (1.0, 0.0),
    "fit_vs_array_att_max_diff": (0.0, 1e-9),   # public .fit() == array path
    "fit_vs_array_se_max_diff": (0.0, 1e-9),
}
