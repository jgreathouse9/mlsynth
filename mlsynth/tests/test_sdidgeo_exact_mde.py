"""The effect a design actually detects, instead of the nearest grid point.

``mde`` is the smallest simulated effect whose power clears the threshold, so
its resolution is the effect grid's step. On a 0.05 grid a design that truly
detects 0.1014 reports 0.150, because 0.1014 misses the 0.10 point and rounds to
the next one -- a 48% overstatement of the lift the experiment needs.

Under placebo inference the detection condition is closed form in the effect
size. Detection is ``2(1 - Phi(|tau| / sigma)) < alpha``, and the analytic
shortcut gives ``tau = tau_0 + e * mean(y_post)``, linear in ``e``. So the
effect where a backtest starts detecting solves directly:

    e_up   = ( z * sigma - tau_0) / mean(y_post)
    e_down = (-z * sigma - tau_0) / mean(y_post)

Two directions, not one, and they are not a symmetric pair. The test rejects
when ``tau`` leaves ``[-z sigma, z sigma]``, so the detection region is
``e < down`` or ``e > up`` with ``down < up`` always -- but nothing forces
``down < 0 < up``. When a backtest's own placebo effect ``tau_0`` is large and
negative, both crossings land positive, and ``down > 0`` says the design rejects
at zero injected effect: it is firing on its own drift. The grid hides that
behind a plausible-looking number, because a small positive effect still sits
inside the rejection region.

With several backtests, power above the threshold means at least
``k = floor(threshold * n_sims) + 1`` of them detect, so the design's boundary
is the k-th smallest per-backtest boundary -- an order statistic, not a mean.
That is why this cannot ride through ``compute_power``, which averages.

``mde`` and every rank column are untouched. They are what the GeoLift
cross-validation pins, and the exact boundary is reported beside them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mlsynth import SDIDGEO
from mlsynth.config_models import SDIDGEOConfig
from mlsynth.utils.datautils import geoex_dataprep
from mlsynth.utils.sdidgeo_helpers.aggregate import compute_exact_mde
from mlsynth.utils.sdidgeo_helpers.engines import resolve_engine
from mlsynth.utils.sdidgeo_helpers.shaping import aggregate_treated, donor_matrix

DATA = Path(__file__).resolve().parents[2] / "basedata" / "geolift_test_data.csv"
ALPHA = 0.1


@pytest.fixture(scope="module")
def panel():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def wide(panel):
    return geoex_dataprep(panel, "location", "date", "Y")["Ywide"]


@pytest.fixture(scope="module")
def arrays(wide):
    region = frozenset(list(wide.columns)[:2])
    y = aggregate_treated(wide, region, how="mean").to_numpy()
    Y0 = donor_matrix(wide, region).to_numpy()
    n_pre = wide.shape[0] - 14
    return y, Y0, n_pre, n_pre, wide.shape[0] - 1, len(region)


# ---------------------------------------------------------------------------
# The engine's boundary
# ---------------------------------------------------------------------------

class TestBoundary:
    def test_matches_a_brute_force_search(self, arrays):
        # The claim the closed form makes: it is the effect at which a fine
        # search would first find the p-value crossing alpha.
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("sdid")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        kw = dict(n_draws=15, n_tr=n_tr, seed=1)
        up, down = eng.detection_boundary(fit, y, Y0, n_pre, start, end,
                                          alpha=ALPHA, **kw)

        grid = np.round(np.arange(0.0, 0.6, 0.0005), 6)
        swept = eng.sweep_p_values(fit, y, Y0, n_pre, start, end, list(grid),
                                   **kw)
        detected = grid[np.asarray(swept["p_value"]) < ALPHA]
        assert detected.size, "no effect on the search grid detected"
        assert up == pytest.approx(float(detected.min()), abs=0.001)

    def test_down_boundary_matches_a_brute_force_search(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("sdid")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        kw = dict(n_draws=15, n_tr=n_tr, seed=1)
        _, down = eng.detection_boundary(fit, y, Y0, n_pre, start, end,
                                         alpha=ALPHA, **kw)
        grid = np.round(np.arange(-0.6, 0.0, 0.0005), 6)
        swept = eng.sweep_p_values(fit, y, Y0, n_pre, start, end, list(grid),
                                   **kw)
        detected = grid[np.asarray(swept["p_value"]) < ALPHA]
        assert detected.size
        assert down == pytest.approx(float(detected.max()), abs=0.001)

    def test_down_is_below_up(self, arrays):
        # The invariant that always holds: the rejection region is the outside
        # of an interval, so the lower crossing sits below the upper one. Their
        # signs are not guaranteed -- a backtest drifting hard enough puts both
        # on the same side, which is what a rejected null at zero effect looks
        # like.
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("sdid")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        up, down = eng.detection_boundary(fit, y, Y0, n_pre, start, end,
                                          alpha=ALPHA, n_draws=15, n_tr=n_tr,
                                          seed=1)
        assert down < up
        assert abs(up) != pytest.approx(abs(down), rel=1e-6)

    def test_a_tighter_alpha_needs_a_bigger_effect(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("sdid")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        kw = dict(n_draws=15, n_tr=n_tr, seed=1)
        lax, _ = eng.detection_boundary(fit, y, Y0, n_pre, start, end,
                                        alpha=0.20, **kw)
        strict, _ = eng.detection_boundary(fit, y, Y0, n_pre, start, end,
                                           alpha=0.01, **kw)
        assert strict > lax

    def test_conformal_reports_no_boundary(self, arrays):
        # The conformal p-value re-permutes against the injected series, so it
        # is not analytic in the effect size. Reported absent instead of guessed.
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("augsynth")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        up, down = eng.detection_boundary(fit, y, Y0, n_pre, start, end,
                                          alpha=ALPHA, inference="conformal",
                                          ns=40, n_tr=n_tr, seed=0)
        assert np.isnan(up) and np.isnan(down)

    def test_augsynth_under_placebo_has_one(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("augsynth")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        up, down = eng.detection_boundary(fit, y, Y0, n_pre, start, end,
                                          alpha=ALPHA, inference="placebo",
                                          n_draws=12, n_tr=n_tr, seed=0)
        assert np.isfinite(up) and np.isfinite(down)
        assert up > 0 > down


# ---------------------------------------------------------------------------
# Aggregating over backtests
# ---------------------------------------------------------------------------

class TestOrderStatistic:
    def _cube(self, ups, downs, duration=7):
        rows = []
        for sim, (u, d) in enumerate(zip(ups, downs), start=1):
            rows.append({"candidate": frozenset({"a"}), "duration": duration,
                         "sim": sim, "boundary_up": u, "boundary_down": d})
        return pd.DataFrame(rows)

    def test_takes_the_kth_smallest_not_the_mean(self):
        # Four backtests at a 0.8 threshold: k = floor(0.8*4)+1 = 4, so every
        # backtest must detect and the boundary is the largest of them.
        cube = self._cube([0.05, 0.08, 0.11, 0.20], [-0.06, -0.09, -0.12, -0.30])
        out = compute_exact_mde(cube, power_threshold=0.8)
        row = out.iloc[0]
        assert row["mde_exact_up"] == pytest.approx(0.20)
        assert row["mde_exact_down"] == pytest.approx(-0.30)

    def test_a_lower_threshold_needs_fewer_backtests(self):
        # k = floor(0.5*4)+1 = 3 -> the third smallest.
        cube = self._cube([0.05, 0.08, 0.11, 0.20], [-0.06, -0.09, -0.12, -0.30])
        out = compute_exact_mde(cube, power_threshold=0.5)
        assert out.iloc[0]["mde_exact_up"] == pytest.approx(0.11)
        assert out.iloc[0]["mde_exact_down"] == pytest.approx(-0.12)

    def test_single_backtest_is_its_own_boundary(self):
        out = compute_exact_mde(self._cube([0.09], [-0.07]),
                                power_threshold=0.8)
        assert out.iloc[0]["mde_exact_up"] == pytest.approx(0.09)
        assert out.iloc[0]["mde_exact_down"] == pytest.approx(-0.07)

    def test_non_finite_boundaries_give_nan(self):
        cube = self._cube([np.nan, np.nan], [np.nan, np.nan])
        out = compute_exact_mde(cube, power_threshold=0.8)
        assert np.isnan(out.iloc[0]["mde_exact_up"])

    def test_empty_cube(self):
        out = compute_exact_mde(pd.DataFrame(), power_threshold=0.8)
        assert out.empty


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

class TestShortlist:
    @pytest.fixture(scope="class")
    def result(self, panel):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SDIDGEO(SDIDGEOConfig(
                df=panel, unitid="location", time="date", outcome="Y",
                treatment_size=2, durations=[14],
                effect_sizes=[-0.2, -0.1, 0.0, 0.1, 0.2], n_backtests=3,
                n_draws=25, n_validation_backtests=0, alpha=ALPHA,
                seed=0)).fit()

    def test_columns_present(self, result):
        for col in ("mde_exact_up", "mde_exact_down", "pre_rmspe_lambda"):
            assert col in result.power.columns, col

    def test_boundaries_are_ordered(self, result):
        s = result.power.dropna(subset=["mde_exact_up", "mde_exact_down"])
        assert not s.empty
        assert (s["mde_exact_down"] < s["mde_exact_up"]).all()

    def test_grid_overstates_where_the_null_is_not_already_rejected(self, result):
        # Restricted to designs that straddle zero, which is the case the grid
        # is meant to describe. There the grid rounds up to its next point, so
        # the exact crossing is at or below it -- and strictly below somewhere,
        # which is the resolution the grid loses.
        s = result.power.dropna(subset=["mde", "mde_exact_up", "mde_exact_down"])
        healthy = s[(s["mde_exact_down"] < 0) & (s["mde_exact_up"] > 0)]
        assert not healthy.empty
        smallest = np.minimum(healthy["mde_exact_up"].abs(),
                              healthy["mde_exact_down"].abs())
        assert (healthy["mde"].abs() + 1e-9 >= smallest).all()
        assert (healthy["mde"].abs() - smallest > 1e-6).any()

    def test_a_rejected_null_shows_as_a_positive_lower_crossing(self, result):
        # down > 0 means zero injected effect already rejects. The grid reports
        # an ordinary-looking MDE for these, so the boundary is the only column
        # that says so.
        s = result.power.dropna(subset=["mde_exact_down"])
        broken = s[s["mde_exact_down"] > 0]
        if not broken.empty:
            assert (broken["mde"].abs() > 0).all()

    def test_lambda_rmse_is_reported_and_usually_lower(self, result):
        # Usually, not always. SDID's time weights solve a problem the treated
        # unit takes no part in: lambda is chosen so a convex combination of the
        # donors' pre-period rows reproduces each donor's post-period mean. The
        # treated unit's lambda-weighted pre-period dispersion is a by-product
        # of that, not a quantity anything optimises, so nothing orders it
        # against the unweighted RMSE. On this panel the ratio runs about 0.72
        # at the median and exceeds 1 for a few designs.
        s = result.power.dropna(subset=["pre_rmspe", "pre_rmspe_lambda"])
        assert not s.empty
        assert (s["pre_rmspe_lambda"] >= 0).all()
        ratio = s["pre_rmspe_lambda"] / s["pre_rmspe"]
        assert ratio.median() < 1.0

    def test_mde_and_ranks_are_unchanged(self, result):
        # Pinned from the run before these columns existed. The GeoLift
        # cross-validation depends on every one of them.
        top = result.power.sort_values("rank").head(5)
        got = [(float(r["rank"]), float(r["mde"]),
                sorted(map(str, r["candidate"]))) for _, r in top.iterrows()]
        assert got == [
            (1.0, 0.2, ["jacksonville", "minneapolis"]),
            (2.0, 0.2, ["atlanta", "nashville"]),
            (3.0, 0.1, ["cleveland", "san francisco"]),
            (3.0, 0.2, ["milwaukee", "orlando"]),
            (5.0, 0.2, ["detroit", "new orleans"]),
        ]
