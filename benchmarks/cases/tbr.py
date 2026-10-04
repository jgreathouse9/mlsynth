"""TBR cross-validation against google/matched_markets.

Cross-validation against the reference implementation, on two panels, because
neither alone reaches everything TBR has to get right.

The GeoLift arm uses ``basedata/geolift_test_data.csv``, which this repository
already ships: 40 named US markets over 105 daily periods, balanced, with no
treatment and no spend. Splitting the markets and naming the window directly
exercises design mode -- the route a candidate design is scored through, and the
only route that works on a panel where nothing happened. It is also an A/A test:
the truth is zero, so the cumulative band has to cover it.

The generated arm reaches what a real untreated panel cannot. Spend confined to
the treated test cells, so the incremental return on ad spend exists and section
3.4's fixed-cost branch fires; a cooldown window, so the intervention and
cooldown halves of the effect are separable; and absent cells concentrated on the
small geos, which is what a panel looks like when a zero-sales day is dropped
instead of recorded. Every constant in the generator is a round design parameter,
chosen so the panel matches the reference panel's shape and hardness -- 100 geos
over 93 periods split evenly, 75 cells absent, a group correlation near 0.996 and
a pretest R-squared near 0.993. Nothing is fitted to any real dataset and no
value from one appears.

Expected values come from ``google/matched_markets`` run on these same two
panels. The reference is Apache 2.0 and is not vendored, so the numbers are
pinned as literals and the case needs nothing beyond this repository. That is
deliberate: a case that skips when the reference is absent is not a gate, and the
regression it guards against is real. Benchmarking the estimator against the
reference is what found two defects the equations alone could not -- a fill that
carried a period flag along the unit axis, and a pretest diagnostic that read a
perfect fit by construction.

Tolerances. Agreement is at 1e-11 relative or better on every quantity, so each
pin is set near 1e-6 of its own magnitude: tight enough that any real change in
the arithmetic fails the case, loose enough to survive a different BLAS.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from mlsynth import TBR
from mlsynth.config_models import TBRConfig

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "geolift_test_data.csv")
_POST_FROM = pd.Timestamp("2021-03-15")

# ---------------------------------------------------------------------------
# the generated panel: round design parameters, matched to the reference's shape
# ---------------------------------------------------------------------------
N_GEOS = 100
N_PRE, N_TEST, N_COOL = 42, 28, 23
START = "2020-01-06"
SIZE_LOG_SD = 1.25
COMMON_SD = 0.75
IDIO_SD = 0.15
LIFT = 0.12
TOTAL_COST = 50_000.0
N_HOLES = 75
SEED = 20200106


def _generated_panel() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    T = N_PRE + N_TEST + N_COOL
    dates = pd.date_range(START, periods=T, freq="D")
    geos = [f"g{i:03d}" for i in range(1, N_GEOS + 1)]

    size = rng.lognormal(0.0, SIZE_LOG_SD, N_GEOS)
    size = 200.0 * size / size.mean()
    size = size[np.argsort(-size)]                 # g001 largest, g100 smallest

    common = 1.0 + COMMON_SD * rng.normal(size=(T, 1))
    idio = 1.0 + IDIO_SD * rng.normal(size=(T, N_GEOS))
    sales = size * (0.5 * common + 0.5 * idio)

    treated = np.zeros(N_GEOS, dtype=bool)
    treated[rng.permutation(N_GEOS)[: N_GEOS // 2]] = True
    post = np.arange(T) >= N_PRE
    sales[np.ix_(post, treated)] *= 1.0 + LIFT

    test = post & (np.arange(T) < N_PRE + N_TEST)
    weights = rng.uniform(0.5, 1.5, size=(T, N_GEOS)) * np.outer(test, treated)
    cost = TOTAL_COST * weights / weights.sum()

    frame = pd.DataFrame({
        "geo": np.repeat(geos, T),
        "date": np.tile(dates, N_GEOS),
        "sales": sales.T.ravel(),
        "cost": cost.T.ravel(),
        "is_control": np.repeat((~treated).astype(int), T),
        "D": (np.tile(post, N_GEOS) & np.repeat(treated, T)).astype(int),
        "cooldown": np.tile(np.arange(T) >= N_PRE + N_TEST, N_GEOS).astype(int),
    })

    weight = np.arange(1, N_GEOS + 1, dtype=float) ** 3.0
    weight[: N_GEOS // 2] = 0.0                    # only the small half
    holes = set()
    for gi in rng.choice(N_GEOS, size=N_HOLES, replace=True,
                         p=weight / weight.sum()):
        while True:
            ti = int(rng.integers(T))
            if (gi, ti) not in holes:
                holes.add((gi, ti))
                break
    mask = np.zeros(len(frame), dtype=bool)
    for gi, ti in holes:
        mask[gi * T + ti] = True
    return frame[~mask].reset_index(drop=True)


def _geolift_panel() -> pd.DataFrame:
    df = pd.read_csv(_DATA, parse_dates=["date"])
    markets = sorted(df.location.unique())
    df["is_treatment"] = df.location.isin(set(markets[0::2])).astype(int)
    df["is_control"] = 1 - df["is_treatment"]
    df["post"] = (df.date >= _POST_FROM).astype(int)
    return df


def run() -> dict:
    warnings.filterwarnings("ignore")

    # --- design mode on the shipped GeoLift panel, which is also an A/A test --
    geo = TBR(TBRConfig(
        df=_geolift_panel(), unitid="location", time="date", outcome="Y",
        control_col="is_control", treatment_col="is_treatment",
        post_col="post", level=0.9, display_graphs=False)).fit()
    g_cum = np.asarray(geo.report.cumulative.estimate, float)

    # --- the generated panel: cost, cooldown, absent cells -------------------
    gen = TBR(TBRConfig(
        df=_generated_panel(), unitid="geo", time="date", outcome="sales",
        treat="D", control_col="is_control", cooldown_col="cooldown",
        cost_col="cost", level=0.9, display_graphs=False)).fit()
    n_cum = np.asarray(gen.report.cumulative.estimate, float)

    return {
        # GeoLift arm
        "geo_alpha": float(geo.report.tbr_fit.alpha),
        "geo_beta": float(geo.report.tbr_fit.beta),
        "geo_sigma_sq": float(geo.report.tbr_fit.sigma_sq),
        "geo_df": float(geo.report.tbr_fit.df),
        "geo_delta_T": float(g_cum[-1]),
        "geo_scale_T": float(geo.report.cumulative.scale[-1]),
        "geo_rmse_pre": float(geo.report.fit_diagnostics.rmse_pre),
        "geo_r2_pre": float(geo.report.fit_diagnostics.r_squared_pre),
        "geo_horizons": float(len(g_cum)),
        # the A/A band has to contain zero: 1.0 when it does
        "geo_covers_zero": float(geo.report.cumulative.lower[-1] <= 0.0
                                 <= geo.report.cumulative.upper[-1]),
        # generated arm
        "gen_alpha": float(gen.report.tbr_fit.alpha),
        "gen_beta": float(gen.report.tbr_fit.beta),
        "gen_sigma_sq": float(gen.report.tbr_fit.sigma_sq),
        "gen_delta_T": float(n_cum[-1]),
        "gen_scale_T": float(gen.report.cumulative.scale[-1]),
        "gen_cum_lower": float(gen.report.cumulative.lower[-1]),
        "gen_iroas": float(gen.report.iroas.estimate),
        "gen_iroas_lower": float(gen.report.iroas.lower),
        "gen_iroas_upper": float(gen.report.iroas.upper),
        "gen_incr_cost": float(gen.report.iroas.total_incremental_cost),
        "gen_filled_cells": float(gen.report.filled_cells),
        "gen_intervention_periods": float(gen.report.intervention_periods),
        "gen_cooldown_periods": float(gen.report.cooldown_periods),
        "gen_fixed_cost": float(gen.report.iroas.fixed_cost),
    }


# Reference values, from google/matched_markets run on these same two panels.
# Every quantity with a reference counterpart agrees to 2.33e-11 relative or
# better, so each is pinned near 1e-6 of its own magnitude. The counts and the
# two indicator metrics are exact and carry a half-unit tolerance.
#
# geo_rmse_pre and geo_r2_pre have no reference counterpart -- the reference
# exposes no pretest fit summary -- so they are pinned at mlsynth's own values.
# They are here because they read 0.0 and 1.0 before the counterfactual carried
# the fitted relation through the pretest, which is the regression this guards.
EXPECTED = {
    # GeoLift panel, design mode, A/A
    "geo_alpha": (3610.656276952813, 0.0036),
    "geo_beta": (0.9946754745387734, 1e-06),
    "geo_sigma_sq": (9892060.484204302, 9.9),
    "geo_df": (71.0, 0.5),
    "geo_delta_T": (7414.601575096211, 0.0074),
    "geo_scale_T": (21375.489914644364, 0.021),
    "geo_rmse_pre": (3101.7809607502386, 0.0031),
    "geo_r2_pre": (0.9771346865000584, 1e-06),
    "geo_horizons": (32.0, 0.5),
    "geo_covers_zero": (1.0, 0.1),
    # generated panel: cost, cooldown, 75 absent cells
    "gen_alpha": (144.5667235446092, 0.00015),
    "gen_beta": (1.1680885588535448, 1.2e-06),
    "gen_sigma_sq": (136122.25349175505, 0.14),
    "gen_delta_T": (64632.42388796776, 0.065),
    "gen_scale_T": (3921.2724806483716, 0.0039),
    "gen_cum_lower": (58029.58524787384, 0.058),
    "gen_iroas": (1.2987107391953314, 1.3e-06),
    "gen_iroas_lower": (1.1660346466826597, 1.2e-06),
    "gen_iroas_upper": (1.431386831708003, 1.4e-06),
    "gen_incr_cost": (49766.60463130796, 0.05),
    "gen_filled_cells": (75.0, 0.5),
    "gen_intervention_periods": (28.0, 0.5),
    "gen_cooldown_periods": (23.0, 0.5),
    "gen_fixed_cost": (1.0, 0.1),
}
