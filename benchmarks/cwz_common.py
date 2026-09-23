"""The Chernozhukov, Wüthrich & Zhu (2021) simulation design, ported.

The DGP behind the JASA paper's Section 4 size and power tables, transcribed
from the authors' own ``sim()`` in the online supplement
(``replication_package_final/simulations_conformal_final.R``). It lives here
because two things need it and neither is the right owner: the Monte Carlo case
draws from it, and the seed-matched comparison needs the same panel layout to
read the reference's dumped draws back in.

The design is a two-factor model with an idiosyncratic AR(1) per unit:

.. math::

    Y^0_{tj} = \\lambda_{1j} + F_{1t} + \\lambda_{2j} F_{2t} + \\varepsilon_{tj},
    \\qquad Y^1_t = (Y^0 w)_t + u_t

with :math:`\\lambda_1 = \\lambda_2 = (1, \\dots, J)/J`, :math:`F_1` standard
normal, :math:`F_2` standard normal in the stationary design and standard normal
plus a linear trend in the non-stationary one, and :math:`\\varepsilon_{\\cdot j}`
and :math:`u` unit-variance AR(1) processes.

The weight vector is the DGP, and what it varies is whether the treated unit is
inside the donor hull:

===== ============================= ==========================================
DGP   :math:`w`                     what it is
===== ============================= ==========================================
1     :math:`1/J` each              uniform, inside the hull
2     :math:`(1/3, 1/3, 1/3, 0 …)`  sparse, inside the hull
3     :math:`-1/J` each             negative uniform, outside the hull
4     :math:`(1, -1, 0, …)`         differencing, outside the hull
===== ============================= ==========================================

DGPs 3 and 4 put the treated unit where a simplex synthetic control cannot
follow it, so the fit is misspecified by construction. The conformal test is
still exact there, which is the paper's claim: its validity rests on the
residual path being exchangeable, not on the control being right.

The random draws are not seed-matched to R's -- the two languages' generators
differ, and matching them is neither possible nor necessary. What is matched is
the design, and the seam where an implementation can go wrong is checked
separately by handing mlsynth the reference's own dumped panels.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

#: The four weight vectors, by DGP index.
DGPS = (1, 2, 3, 4)


def dgp_weights(dgp: int, n_donors: int) -> np.ndarray:
    """The treated unit's weight vector under ``dgp`` (supplement ``sim()``)."""
    if dgp not in DGPS:
        raise ValueError(f"DGP must be one of {DGPS}; got {dgp!r}.")
    w = np.zeros(n_donors)
    if dgp == 1:
        w[:] = 1.0 / n_donors
    elif dgp == 2:
        w[:3] = 1.0 / 3.0
    elif dgp == 3:
        w[:] = -1.0 / n_donors
    else:
        w[0], w[1] = 1.0, -1.0
    return w


def ar1_unit_variance(n_periods: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance AR(1), started from its own stationary distribution.

    The supplement's ``generate.AR.series``: innovations scaled by
    ``sqrt(1 - rho^2)`` so the unconditional variance is one whatever ``rho``,
    and a standard-normal start value, so the series is stationary from ``t=1``
    and the pre-period is not systematically quieter than the post-period.
    """
    innovations = rng.standard_normal(n_periods) * np.sqrt(1.0 - rho ** 2)
    x = np.empty(n_periods)
    x[0] = rho * rng.standard_normal() + innovations[0]
    for t in range(1, n_periods):
        x[t] = rho * x[t - 1] + innovations[t]
    return x


def simulate_conformal_panel(
    dgp: int,
    *,
    pre_periods: int,
    post_periods: int,
    n_donors: int,
    rho_donor: float,
    rho_treated: float,
    stationary: bool,
    alternative: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """One draw from the design. Returns ``(y, Y0)`` of shapes ``(T,)``, ``(T, J)``."""
    n_periods = pre_periods + post_periods
    f1 = rng.standard_normal(n_periods)
    f2 = rng.standard_normal(n_periods)
    if not stationary:
        f2 = f2 + np.arange(1, n_periods + 1, dtype=float)
    lam = np.arange(1, n_donors + 1, dtype=float) / n_donors
    signal = lam[None, :] + f1[:, None] + np.outer(f2, lam)
    eps = np.column_stack(
        [ar1_unit_variance(n_periods, rho_donor, rng) for _ in range(n_donors)]
    )
    Y0 = signal + eps
    y = Y0 @ dgp_weights(dgp, n_donors) + ar1_unit_variance(n_periods, rho_treated, rng)
    y[pre_periods:] += alternative
    return y, Y0


def moving_block_pvalue(y: np.ndarray, Y0: np.ndarray, pre_periods: int) -> float:
    """The moving-block conformal p-value for the null of no effect.

    Goes through the same public seam the estimator uses -- the ``"sc"`` refit
    rule and the cyclic-shift reference set -- so what this measures is what
    ``VanillaSC(inference="conformal", conformal_type="block")`` measures, at a
    fraction of the cost of building a DataFrame per draw.
    """
    from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue

    return float(
        conformal_pvalue(y, Y0, pre_periods, conformal_type="block", refit="sc")
    )


def load_reference_draws(path: str) -> list:
    """Read a ``draws_*.tsv`` dumped by the reference run into ``(y, Y0)`` pairs."""
    import pandas as pd

    table = pd.read_csv(path, sep="\t")
    donor_cols = [c for c in table.columns if c.startswith("Y0_")]
    return [
        (block["Y1"].to_numpy(dtype=float), block[donor_cols].to_numpy(dtype=float))
        for _, block in table.groupby("draw", sort=True)
    ]


# ---------------------------------------------------------------------------
# The t-test paper's application-calibrated design (JPE package: the
# calibration in calibration_dgps.R, the draw in common_functions.R)
# ---------------------------------------------------------------------------

#: The nine DGPs of Table 3, built by :func:`ttest_dgp_spec`. What varies across
#: them is whether a simplex synthetic control can span the treated unit (DGP 1
#: and 2 yes, the rest use the paper's misspecified ``(-3, 3, 1, 0 …)``) and how
#: the panel departs from stationarity -- nothing, a common linear trend, a
#: common random walk, a common trend with one donor deviating, or a
#: donor-specific slope. The treatment effect is zero throughout, so coverage of
#: a nominal interval and the bias of the debiased estimator are what the design
#: measures.
TTEST_DGPS = tuple(range(1, 10))


def ttest_dgp_spec(dgp: int, n_periods: int, n_donors: int, w_sc, rng):
    """``(w, const, nonstat)`` for one of the nine DGPs (``common_functions.R``)."""
    w_did = np.full(n_donors, 1.0 / n_donors)
    w_mis = np.zeros(n_donors)
    w_mis[:3] = (-3.0, 3.0, 1.0)
    t = np.arange(1, n_periods + 1, dtype=float)
    const, nonstat = 0.0, np.zeros((n_periods, n_donors))
    if dgp == 1:
        w = np.asarray(w_sc, dtype=float)
    elif dgp == 2:
        w = w_did
    elif dgp == 3:
        w, const = w_mis, 2.0
    elif dgp == 4:
        w, const = w_mis, 2.0
        nonstat = np.tile(t[:, None], (1, n_donors))
    elif dgp == 5:
        w, const = w_mis, 2.0
        nonstat = np.tile(np.cumsum(rng.standard_normal(n_periods))[:, None],
                          (1, n_donors))
    elif dgp == 6:
        w = np.asarray(w_sc, dtype=float)
        nonstat = np.tile(t[:, None], (1, n_donors))
        nonstat[:, 0] = nonstat[:, 0] + t
    elif dgp == 7:
        w = np.asarray(w_sc, dtype=float)
        nonstat = np.tile(np.cumsum(rng.standard_normal(n_periods))[:, None],
                          (1, n_donors))
        nonstat[:, 0] = nonstat[:, 0] + np.cumsum(rng.standard_normal(n_periods))
    elif dgp == 8:
        w = w_mis
        j = np.arange(1, n_donors + 1, dtype=float)
        nonstat = j[None, :] + np.outer(t, j)
    elif dgp == 9:
        w = np.asarray(w_sc, dtype=float)
        nonstat = np.empty((n_periods, n_donors))
        for jj in range(n_donors):
            noise = rng.standard_normal(n_periods)
            theta = np.empty(n_periods)
            theta[0] = (jj + 1) + noise[0]
            for tt in range(1, n_periods):
                theta[tt] = (jj + 1) + theta[tt - 1] + noise[tt]
            nonstat[:, jj] = theta
    else:
        raise ValueError(f"DGP must be 1..9; got {dgp!r}.")
    return w, const, nonstat


def ar1_given_variance(n_periods, rho, var_error, rng):
    """AR(1) with a given innovation variance (``generate.AR.series.var.input``)."""
    innovations = rng.standard_normal(n_periods) * np.sqrt(var_error)
    start = rng.standard_normal() * np.sqrt(var_error / (1.0 - rho ** 2))
    x = np.empty(n_periods)
    x[0] = rho * start + innovations[0]
    for t in range(1, n_periods):
        x[t] = rho * x[t - 1] + innovations[t]
    return x


def simulate_ttest_panel(dgp, calib, *, pre_periods, post_periods, rng):
    """One draw from the calibrated design. Returns ``(y, Y0, w_true)``."""
    n_periods = pre_periods + post_periods
    n_donors = calib["Lambda"].shape[0]
    w, const, nonstat = ttest_dgp_spec(dgp, n_periods, n_donors, calib["w_sc"], rng)
    factors = rng.standard_normal((n_periods, calib["var_factors"].size)) * np.sqrt(
        calib["var_factors"])[None, :]
    eps = np.column_stack([
        ar1_given_variance(n_periods, calib["rho"][j], calib["var_epsl"][j], rng)
        for j in range(n_donors)
    ])
    Y0 = nonstat + factors @ calib["Lambda"].T + eps
    y = const + Y0 @ w + ar1_given_variance(
        n_periods, calib["rho_u"], calib["var_u"], rng)
    return y, Y0, w


def load_ttest_calibration(ref_dir: str, rho_u: float, var_u: float) -> dict:
    """Read the calibration the reference run fitted and wrote out.

    The calibration is an input to the design, not the thing under test, so it
    is taken from the reference instead of re-derived. R's ``ar()`` picks its
    order by AIC and solves Yule-Walker; re-implementing that in Python would
    insert a second estimator between the two sides and buy nothing.
    """
    import os

    import pandas as pd

    units = pd.read_csv(os.path.join(ref_dir, "calibration_units.tsv"), sep="\t")
    lam = pd.read_csv(os.path.join(ref_dir, "calibration_lambda.tsv"), sep="\t")
    fac = pd.read_csv(os.path.join(ref_dir, "calibration_factors.tsv"), sep="\t")
    return {
        "w_sc": units["w0_sc"].to_numpy(dtype=float),
        "rho": units["rho"].to_numpy(dtype=float),
        "var_epsl": units["var_epsl"].to_numpy(dtype=float),
        "Lambda": lam[[c for c in lam.columns if c.startswith("lambda_")]]
                    .to_numpy(dtype=float),
        "var_factors": fac["var"].to_numpy(dtype=float),
        "rho_u": float(rho_u),
        "var_u": float(var_u),
    }


def oracle_ttest(y, Y0, pre_periods, post_periods, w_true, n_folds):
    """The paper's oracle benchmark: known weights, no estimation (``oracle.cf``)."""
    block = min(pre_periods // n_folds, post_periods)
    post_gap = float(np.mean(y[pre_periods:] - Y0[pre_periods:] @ w_true))
    taus = np.empty(n_folds)
    for k in range(n_folds):
        start = pre_periods - block * n_folds + k * block
        idx = slice(start, start + block)
        taus[k] = post_gap - float(np.mean(y[idx] - Y0[idx] @ w_true))
    se = np.sqrt(1.0 + (n_folds * block) / post_periods) * taus.std(ddof=1) / np.sqrt(n_folds)
    return float(taus.mean()), float(se)
