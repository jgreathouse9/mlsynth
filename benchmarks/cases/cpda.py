"""CPDA: recovery on Hsiao and Zhou's own data-generating process, and the
slope step cross-checked against the algebra it is defined by.

Path B (the paper's simulation design) plus a closed-form cross-validation.
The empirical route, Path A, is deliberately not the pin here, and why is the
point of this case.

Hsiao and Zhou's Table 9 reports a CPDA mean absolute effect of 9.56 on the
California cigarette panel. That number is not reproducible from the paper's
description, and the finding is a property of that description, not a defect
in this implementation. Equations 11 to 15 fix every step except which
control units enter, and the paper says only that the subset "can be chosen
using a model selection criterion as in Hsiao, Ching, and Wan (2012), or the
LASSO method ... as suggested by Li and Bell (2017)". Measured on that panel,
19 pre-treatment years against 38 controls, four defensible readings of that
sentence span a mean absolute effect of 3.79 to 14.04.

Under leave-one-pre-period-out error with the selection repeated inside every
fold, the lowest error belongs to the selector giving 4.93, and the readings
landing nearest 9.56 score worst. So no criterion computable from the
pre-period picks the published value out, and pinning this case to 9.56 would
pin a number reached by tuning against the post-period. The study behind that
measurement is ``benchmarks/studies/hsiao_zhou_counterfactuals``.

What is pinned instead:

* recovery -- on the paper's Equation 2 to 3 design with a planted effect, the
  ATT's mean absolute error over ten seeds;
* the covariate step earning its place -- the same panels fit by a donor-only
  LASSO, whose error is a multiple of CPDA's when the treated unit's covariate
  moves in a way no control spans;
* the slope -- ``beta_cce`` against Pesaran's Equation 16 computed directly,
  and ``beta_bai`` against the objective it is the argmin of;
* the selector spread -- that ``sensitivity=True`` reports one, and that it
  brackets the point estimate, since an estimator that hid this would claim an
  identification the method does not have.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

N_SEEDS = 10
TRUTH = -5.0


def _panel(seed: int, n_donors: int = 12, T: int = 40, T0: int = 30,
           treated_sd: float = 8.0, beta=(2.0, -1.0)) -> pd.DataFrame:
    """Equations 2 and 3 with two covariates, one treated unit, two factors.

    ``treated_sd`` gives the treated unit's first covariate a swing of its own,
    independent of every control's, so the control outcomes cannot span it. The
    controls still vary in the same covariate, which is what identifies the
    slope: it is estimated from the control group alone.
    """
    rng = np.random.default_rng(seed)
    N = n_donors + 1
    f = rng.standard_normal((T, 2))
    gam = rng.standard_normal((N, 2))
    x0 = 10.0 + rng.standard_normal((T, N))
    x0[:, 0] = 10.0 + treated_sd * rng.standard_normal(T)
    x1 = rng.standard_normal((T, N)) + np.linspace(0, 2, T)[:, None]
    Y = (beta[0] * x0 + beta[1] * x1 + f @ gam.T
         + 0.5 * rng.standard_normal((T, N)))
    Y[T0:, 0] += TRUTH
    units = np.repeat(np.arange(N), T)
    times = np.tile(np.arange(T), N)
    return pd.DataFrame({
        "unit": units, "time": times, "y": Y.T.ravel(),
        "x0": x0.T.ravel(), "x1": x1.T.ravel(),
        "D": ((units == 0) & (times >= T0)).astype(int),
    })


def _cfg(df, **kw):
    base = dict(df=df, outcome="y", treat="D", unitid="unit", time="time",
                covariates=["x0", "x1"], display_graphs=False)
    base.update(kw)
    return base


def run() -> dict:
    from mlsynth import CPDA, PDA
    from mlsynth.utils.cpda_helpers.beta import bai_objective, beta_bai, beta_cce

    # --- recovery, and the covariate step against a donor-only fit ----------
    errs, ratios = [], []
    for seed in range(N_SEEDS):
        df = _panel(seed)
        att = CPDA(_cfg(df)).fit().effects.att
        errs.append(abs(att - TRUTH))
        pda_att = PDA(dict(df=df, outcome="y", treat="D", unitid="unit",
                           time="time", method="LASSO",
                           display_graphs=False)).fit().effects.att
        ratios.append(abs(pda_att - TRUTH) / max(abs(att - TRUTH), 1e-9))

    # --- the slope, against the algebra it is defined by --------------------
    rng = np.random.default_rng(0)
    T0, N, k = 25, 8, 2
    Yc = rng.standard_normal((T0, N))
    Xc = rng.standard_normal((T0, N, k))
    zbar = np.column_stack([Yc.mean(axis=1), Xc.mean(axis=1)])
    Q, _ = np.linalg.qr(zbar)
    M = np.eye(T0) - Q @ Q.T
    A = sum(Xc[:, i, :].T @ M @ Xc[:, i, :] for i in range(N))
    b = sum(Xc[:, i, :].T @ M @ Yc[:, i] for i in range(N))
    cce_gap = float(np.max(np.abs(
        beta_cce(Yc, Xc, T0) - np.linalg.lstsq(A, b, rcond=None)[0])))

    rng = np.random.default_rng(1)
    T, N = 40, 15
    lam = rng.standard_normal((N, 2))
    f = rng.standard_normal((T, 2))
    Xb = np.empty((T, N, 2))
    Xb[:, :, 0] = 10.0 + 0.1 * rng.standard_normal((T, N))
    Xb[:, :, 1] = f @ lam.T + rng.standard_normal((T, N))
    Yb = (np.einsum("tnk,k->tn", Xb, np.array([20.0, 2.0])) + f @ lam.T
          + rng.standard_normal((T, N)))
    ols = np.linalg.lstsq(Xb.reshape(-1, 2), Yb.reshape(-1), rcond=None)[0]
    bai_improvement = float(bai_objective(Yb, Xb, ols, 2)
                            - bai_objective(Yb, Xb, beta_bai(Yb, Xb, r=2), 2))

    # --- the selector spread the estimator reports --------------------------
    res = CPDA(_cfg(_panel(0), sensitivity=True)).fit()
    spread = res.fit.sensitivity
    atts = [v["att"] for v in spread.values()]
    brackets = float(min(atts) - 1e-8 <= res.effects.att <= max(atts) + 1e-8)

    return {
        "recovery_mean_abs_error": float(np.mean(errs)),
        "recovery_max_abs_error": float(np.max(errs)),
        "donor_only_error_ratio_min": float(np.min(ratios)),
        "cce_matches_equation_16": cce_gap,
        "bai_improves_on_pooled_ols": bai_improvement,
        "sensitivity_selectors_reported": float(len(spread)),
        "sensitivity_brackets_point_estimate": brackets,
    }


EXPECTED = {
    # Measured at 0.109 mean and 0.288 max over the ten seeds. The tolerances
    # are the headroom a different BLAS needs, not a slack budget: the planted
    # effect is -5, so even the loose end stays under a tenth of it.
    "recovery_mean_abs_error": (0.109, 0.15),
    "recovery_max_abs_error": (0.288, 0.20),
    # A donor-only LASSO's error is 7.4 times CPDA's at worst over the same
    # seeds and 37 times at the median. Pinned at the worst case with room,
    # since this is what says the covariate step is doing the work.
    "donor_only_error_ratio_min": (7.4, 4.0),
    # Closed form against closed form; the only gap is floating point.
    "cce_matches_equation_16": (0.0, 1e-10),
    # Bai's estimator is the argmin, so its objective cannot exceed pooled
    # OLS's. Measured at 47.5 on this design. The value itself is design
    # specific, so the tolerance is set to assert the sign with margin and
    # nothing narrower.
    "bai_improves_on_pooled_ols": (47.5, 47.0),
    "sensitivity_selectors_reported": (4.0, 0.0),
    "sensitivity_brackets_point_estimate": (1.0, 0.0),
}
