"""The SDIDGEO engine seam: a pluggable scoring estimator.

Everything in SDIDGEO except the fit and the p-value is estimator-independent --
candidate nomination, the backtest windows, effect injection, power, MDE,
ranking, the constraint layer, the plots. This module pins the seam that makes
that explicit, so a second engine can be added without the pipeline noticing.

Two contracts, not one:

* the engine fits, ``fit_once(y, Y0, n_pre, start, end, n_tr) -> EngineFit``;
* the inference produces p-values for a whole effect sweep at once.

The sweep is the inference's job because the two procedures differ in what they
can hoist. The placebo standard error is invariant to the injected effect, so it
is computed once and reused across the grid; a conformal p-value re-permutes
against the treated series and so has to be recomputed per effect size. Handing
the inference the whole grid keeps that choice inside the module that owns it.

Phase 1 adds the seam with ``sdid`` as the only engine, so it is proved by
things not moving: the pipeline's output has to be bit-identical to the version
before the seam existed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlsynth import SDIDGEO
from mlsynth.config_models import SDIDGEOConfig
from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.sdidgeo_helpers.engines import EngineFit, resolve_engine
from mlsynth.utils.sdidgeo_helpers.shaping import aggregate_treated, donor_matrix

DATA = Path(__file__).resolve().parents[2] / "basedata" / "geolift_test_data.csv"


@pytest.fixture(scope="module")
def panel():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def wide(panel):
    from mlsynth.utils.datautils import geoex_dataprep
    return geoex_dataprep(panel, "location", "date", "Y")["Ywide"]


@pytest.fixture(scope="module")
def arrays(wide):
    region = frozenset(list(wide.columns)[:2])
    y = aggregate_treated(wide, region, how="mean").to_numpy()
    Y0 = donor_matrix(wide, region).to_numpy()
    n_pre = wide.shape[0] - 14
    return y, Y0, n_pre, n_pre, wide.shape[0] - 1, len(region)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class TestResolution:
    def test_resolves_sdid(self):
        eng = resolve_engine("sdid")
        assert eng.name == "sdid"
        assert callable(eng.fit_once) and callable(eng.sweep_p_values)
        assert callable(eng.att) and callable(eng.point_inference)

    def test_unknown_engine_raises(self):
        with pytest.raises(MlsynthConfigError, match="engine"):
            resolve_engine("granny-smith")

    def test_config_defaults_to_sdid(self, panel):
        cfg = SDIDGEOConfig(df=panel, unitid="location", time="date",
                            outcome="Y", treatment_size=2, durations=[14],
                            effect_sizes=[0.0, 0.2], n_backtests=2, n_draws=5,
                            seed=0)
        assert cfg.engine == "sdid"

    def test_config_rejects_an_unknown_engine(self, panel):
        with pytest.raises(MlsynthConfigError, match="engine"):
            SDIDGEOConfig(df=panel, unitid="location", time="date", outcome="Y",
                          treatment_size=2, durations=[14],
                          effect_sizes=[0.0, 0.2], n_backtests=2, n_draws=5,
                          seed=0, engine="mcintosh")


# ---------------------------------------------------------------------------
# The EngineFit contract
# ---------------------------------------------------------------------------

class TestEngineFitContract:
    def test_fields_present_and_shaped(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        fit = resolve_engine("sdid").fit_once(y, Y0, n_pre, start, end, n_tr)
        assert isinstance(fit, EngineFit)
        assert fit.counterfactual.shape == y.shape
        assert fit.donor_weights.shape == (Y0.shape[1],)
        assert np.isfinite(fit.pre_rmspe) and np.isfinite(fit.scaled_l2)
        assert isinstance(fit.extras, dict)

    def test_sdid_carries_time_weights(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        fit = resolve_engine("sdid").fit_once(y, Y0, n_pre, start, end, n_tr)
        assert fit.time_weights is not None
        assert fit.time_weights.shape == (n_pre,)
        assert fit.time_weights.min() >= -1e-9
        assert fit.time_weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_donor_weights_are_a_simplex(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        fit = resolve_engine("sdid").fit_once(y, Y0, n_pre, start, end, n_tr)
        assert fit.donor_weights.min() >= -1e-9
        assert fit.donor_weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_att_is_the_mean_window_gap(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("sdid")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        expected = float(np.mean(y[start:end + 1]
                                 - fit.counterfactual[start:end + 1]))
        assert eng.att(fit, y, start, end) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# The seam is inert: it must reproduce the direct engine calls exactly
# ---------------------------------------------------------------------------

class TestSeamIsInert:
    def test_fit_matches_sdid_fit_once(self, arrays):
        from mlsynth.utils.sdidgeo_helpers.engine import sdid_fit_once
        y, Y0, n_pre, start, end, n_tr = arrays
        direct = sdid_fit_once(y, Y0, n_pre, start, end, n_tr=n_tr)
        through = resolve_engine("sdid").fit_once(y, Y0, n_pre, start, end, n_tr)
        np.testing.assert_array_equal(through.counterfactual, direct.counterfactual)
        np.testing.assert_array_equal(through.donor_weights, direct.omega)
        np.testing.assert_array_equal(through.time_weights, direct.lam)

    def test_sweep_matches_the_hoisted_placebo_computation(self, arrays):
        # The placebo sigma is invariant to the injected effect, so the sweep
        # must reuse one draw across the grid, exactly as the loop did before.
        from mlsynth.utils.sdidgeo_helpers.engine import (
            normal_p_value, placebo_sigma, sdid_att, sdid_fit_once)
        y, Y0, n_pre, start, end, n_tr = arrays
        grid = [-0.2, -0.1, 0.0, 0.1, 0.2]

        direct_fit = sdid_fit_once(y, Y0, n_pre, start, end, n_tr=n_tr)
        tau0 = sdid_att(direct_fit, y, start, end)
        baseline = float(np.mean(y[start:end + 1]))
        sigma = placebo_sigma(y, Y0, n_pre, start, end, n_draws=12, n_tr=n_tr,
                              seed=4)
        expected = [normal_p_value(tau0 + es * baseline, sigma) for es in grid]

        eng = resolve_engine("sdid")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        got = eng.sweep_p_values(fit, y, Y0, n_pre, start, end, grid,
                                 n_draws=12, n_tr=n_tr, seed=4)
        np.testing.assert_array_equal(np.asarray(got["p_value"]), expected)
        np.testing.assert_array_equal(np.asarray(got["tau"]),
                                      [tau0 + es * baseline for es in grid])

    def test_point_inference_matches_placebo_plus_normal_test(self, arrays):
        from mlsynth.utils.sdidgeo_helpers.engine import (
            normal_p_value, placebo_sigma, sdid_att, sdid_fit_once)
        y, Y0, n_pre, start, end, n_tr = arrays
        direct_fit = sdid_fit_once(y, Y0, n_pre, start, end, n_tr=n_tr)
        tau = sdid_att(direct_fit, y, start, end)
        sigma = placebo_sigma(y, Y0, n_pre, start, end, n_draws=9, n_tr=n_tr,
                              seed=1)
        eng = resolve_engine("sdid")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        p, details = eng.point_inference(fit, y, Y0, n_pre, start, end,
                                         n_draws=9, n_tr=n_tr, seed=1)
        assert p == normal_p_value(tau, sigma)
        assert details["sigma"] == sigma
        assert details["method"] == "placebo"


# ---------------------------------------------------------------------------
# End to end: the pipeline's answer must not move
# ---------------------------------------------------------------------------

class TestPipelineUnchanged:
    """Pinned from the run immediately before the seam was introduced."""

    @pytest.fixture(scope="class")
    def result(self, panel):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SDIDGEO(SDIDGEOConfig(
                df=panel, unitid="location", time="date", outcome="Y",
                treatment_size=2, durations=[14],
                effect_sizes=[-0.2, -0.1, 0.0, 0.1, 0.2], n_backtests=3,
                n_draws=25, n_validation_backtests=0, seed=0)).fit()

    def test_same_region_selected(self, result):
        assert result.selected_units == ["jacksonville", "minneapolis"]

    def test_same_top_of_the_shortlist(self, result):
        top = result.power.sort_values("rank").head(5)
        got = [(float(r["rank"]), float(r["mde"]), sorted(map(str, r["candidate"])))
               for _, r in top.iterrows()]
        assert got == [
            (1.0, 0.2, ["jacksonville", "minneapolis"]),
            (2.0, 0.2, ["atlanta", "nashville"]),
            (3.0, 0.1, ["cleveland", "san francisco"]),
            (3.0, 0.2, ["milwaukee", "orlando"]),
            (5.0, 0.2, ["detroit", "new orleans"]),
        ]

    def test_same_weights(self, result):
        w = result.design_weights
        assert len(w.donor_weights) == 38 and len(w.time_weights) == 91
        assert w.donor_weights["atlanta"] == pytest.approx(0.01803435933762521,
                                                           abs=1e-12)
        assert w.donor_weights["austin"] == pytest.approx(0.037770064501080655,
                                                          abs=1e-12)


class TestNonAnalyticSweep:
    """The refit path, for engines or checks that cannot use the shortcut.

    ``analytic=True`` shifts the ATT by ``effect_size x mean(treated_post)``
    instead of refitting, which is exact because the fit uses pre-period data
    alone. ``analytic=False`` injects and recomputes, and the two must agree --
    that equality is what licenses the shortcut.
    """

    def test_refit_path_matches_the_analytic_shortcut(self, arrays):
        y, Y0, n_pre, start, end, n_tr = arrays
        eng = resolve_engine("sdid")
        fit = eng.fit_once(y, Y0, n_pre, start, end, n_tr)
        grid = [-0.15, 0.0, 0.15]
        kw = dict(n_draws=8, n_tr=n_tr, seed=2)
        fast = eng.sweep_p_values(fit, y, Y0, n_pre, start, end, grid,
                                  analytic=True, **kw)
        slow = eng.sweep_p_values(fit, y, Y0, n_pre, start, end, grid,
                                  analytic=False, **kw)
        np.testing.assert_allclose(slow["tau"], fast["tau"], rtol=1e-10)
        np.testing.assert_allclose(slow["p_value"], fast["p_value"], rtol=1e-10)
